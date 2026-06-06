"""
phys_hgnet.py — PhysHGNet: Physics-aware Hierarchical Graph Neural Operator.

Extends StructuredDGNet with three innovations:
  C1 - PhysicsAwareAnchorSelector (residual+gradient-weighted FPS)
  C2 - LearnableCoarseOperator (GSL coarse operator)
  C3 - DualScaleGNNCorrector (dual-scale GNN with virtual nodes)

All innovations are individually toggleable via config for ablation studies.
Interface: forward(batch) -> {'u_final': (B, T, N, C)}

================================================================================
WHAT CHANGED vs the previous version (and WHY)
================================================================================
The previous implementation cached the *whole* graph (including anchor indices,
R/P interpolation, coarse graph) by the key (N, device, use_physics_anchor) and
built it exactly once — on the FIRST forward call. At that moment
`self._residual_cache` and `self._grad_norm_cache` were still None (they are only
filled *inside* the time loop, which runs AFTER the graph is built). Two direct
consequences:

  BUG-A  Anchors NEVER updated across time steps. The residual/gradient caches
         were dutifully recomputed every `residual_update_freq` steps, but the
         anchors that depend on them were never re-selected, so they stayed
         frozen for the entire rollout (and for the entire training run).

  BUG-B  C1 (physics-aware anchor selection) was effectively INERT. Because the
         single anchor build happened with residual=None and grad_norm=None, the
         selector always fell back to plain geometric FPS regardless of
         use_physics_anchor=True. The learnable (alpha,beta,gamma) weights had no
         effect on which nodes were chosen.

Fix:
  * The graph cache is split into a STATIC part (depends only on the mesh:
    edges, edge features, local Laplacian weights, Jacobi preconditioner) and a
    DYNAMIC part (anchor indices + R/P + coarse graph) that is rebuilt during the
    rollout using the up-to-date residual/gradient signals.
  * Before the loop we seed the physics caches from the initial condition so the
    very first anchor selection is already physics-aware (fixes BUG-B).
  * Inside the loop, every `residual_update_freq` steps we refresh the caches and
    RE-SELECT the anchors (fixes BUG-A). Controlled by config["dynamic_anchors"].
  * `record_anchors=True` records the anchor index set at every update into
    `self.anchor_history` so it can be visualised (see visualize_anchors.py).

New config keys (all optional, sensible defaults):
  dynamic_anchors   : bool  (default True)  -> re-select anchors during rollout
  anchor_cap_ratio  : int   (default 16)    -> m_target <= N // anchor_cap_ratio
  anchor_cap_max    : int   (default 256)   -> hard upper bound on #anchors

Note (known limitation, unchanged): anchor selection uses argmax (hard FPS) and
is therefore non-differentiable, so the (alpha,beta,gamma) logits receive no
gradient. They still control selection through their *initial* values and any
manual schedule. Making the selection differentiable (e.g. Gumbel-softmax /
soft-FPS) is left as future work; it is orthogonal to the two bugs fixed here.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from structured_models import (
    FineGraphEncoder, build_bidirectional_edges, build_edge_features, build_knn_graph,
    farthest_point_sampling,
)
from structured_dgnet import ImplicitCGSolve, _precond_cg_solve, ResidualSolver as _OrigResidualSolver
from phys_hgnet_modules import (
    PhysicsAwareAnchorSelector, LearnableCoarseOperator, DualScaleGNNCorrector,
    build_restriction_prolongation, build_coarse_edge_attr,
)
from gradient_utils import compute_gradient_norm


DEFAULT_CONFIG: Dict[str, Any] = {
    "spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
    "m_anchors": 64, "q_local": 4, "k_coarse": 6,
    "operator_hidden_dim": 64, "operator_num_layers": 3,
    "residual_hidden_dim": 128, "residual_num_layers": 5,
    "coarse_num_layers": 4, "k_virtual_nodes": 4,
    "cg_max_iter": 50, "cg_tol": 1e-6,
    "use_physics_anchor": True,
    "use_learned_coarse": True,
    "use_dual_scale_gnn": True,
    "use_virtual_nodes": True,
    "residual_update_freq": 5,
    "operator_type": "laplace",
    # ── new keys ──────────────────────────────────────────────
    "dynamic_anchors": True,      # re-select anchors during the rollout
    "anchor_cap_ratio": 16,       # m_target <= N // anchor_cap_ratio
    "anchor_cap_max": 256,        # absolute cap on number of anchors
}


def _sparse_Lv(v, Lp_w, fine_ei):
    """Self-contained sparse Laplacian matvec: (Lv)_i = sum_j w_ij (v_j - v_i)."""
    src, dst = fine_ei
    if v.dim() == 1:
        msg = Lp_w * (v[dst] - v[src])
        out = torch.zeros_like(v)
        out.scatter_add_(0, src, msg)
    else:
        msg = Lp_w.unsqueeze(1) * (v[dst] - v[src])
        out = torch.zeros_like(v)
        idx = src.unsqueeze(1).expand(-1, v.shape[1])
        out.scatter_add_(0, idx, msg)
    return out


def _build_jacobi_precond(L_local_weights, fine_ei, N, dt):
    diag_L = torch.zeros(N, device=L_local_weights.device, dtype=L_local_weights.dtype)
    diag_L.scatter_add_(0, fine_ei[0], L_local_weights)
    diag_A = 1.0 + (dt / 2.0) * diag_L
    return 1.0 / diag_A.clamp(min=1e-8)


def _match_weights_to_edge_index(sp_ei, sp_w, fine_ei, N):
    device = sp_ei.device
    E = fine_ei.shape[1]
    sp_key = sp_ei[0] * N + sp_ei[1]
    fine_key = fine_ei[0] * N + fine_ei[1]
    sorted_key, perm = sp_key.sort()
    sorted_w = sp_w[perm]
    idx = torch.searchsorted(sorted_key, fine_key).clamp(max=sorted_key.shape[0] - 1)
    found = sorted_key[idx] == fine_key
    weights = torch.zeros(E, device=device, dtype=sp_w.dtype)
    weights[found] = sorted_w[idx[found]]
    return weights


def _apply_bcs(u, boundary_info):
    if not isinstance(boundary_info, dict):
        return u
    di = boundary_info.get("dirichlet")
    if di is None:
        return u
    idx = di.get("indices")
    val = di.get("values")
    if idx is None:
        return u
    device = u.device
    idx = idx.to(device)
    if val is not None:
        val = val.to(device)
    u = u.clone()
    if val is not None and idx.numel() > 0:
        u[:, idx] = val.unsqueeze(0).unsqueeze(-1).expand(u.shape[0], -1, u.shape[-1])
    return u


class PhysHGNet(nn.Module):
    """
    Physics-aware Hierarchical Graph Neural Operator.

    Key improvements over DGNet:
      1. CG solver (not dense LU): O(N) memory vs O(N^2), better scalability.
      2. Hierarchical multiscale structure (fine + coarse graphs).
      3. C1: Physics residual + gradient-weighted anchor selection (3-term FPS),
         now actually time-varying (see module docstring).
      4. C2: Learnable coarse-grid operator (data-driven).
      5. C3: Dual-scale GNN with virtual nodes (long-range interactions).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **config}
        self.cfg = cfg

        self.spatial_dim = cfg["spatial_dim"]
        self.feature_dim = cfg["feature_dim"]
        self.output_dim = cfg["output_dim"]
        self.m_anchors = cfg["m_anchors"]
        self.q_local = cfg["q_local"]
        self.k_coarse = cfg["k_coarse"]
        self.cg_max_iter = cfg["cg_max_iter"]
        self.cg_tol = cfg["cg_tol"]
        self.res_upd_freq = max(1, int(cfg["residual_update_freq"]))

        self.use_physics_anchor = cfg["use_physics_anchor"]
        self.use_learned_coarse = cfg["use_learned_coarse"]
        self.use_dual_scale_gnn = cfg["use_dual_scale_gnn"]
        self.use_virtual_nodes = cfg["use_virtual_nodes"]

        # ── new behaviour switches ───────────────────────────
        self.dynamic_anchors = bool(cfg["dynamic_anchors"])
        self.anchor_cap_ratio = max(1, int(cfg["anchor_cap_ratio"]))
        self.anchor_cap_max = int(cfg["anchor_cap_max"])

        op_hd = cfg["operator_hidden_dim"]
        op_nl = cfg["operator_num_layers"]
        self.fine_encoder = FineGraphEncoder(self.spatial_dim, op_hd, op_nl)

        init_raw = math.log(max(math.exp(0.001) - 1.0, 1e-10))
        self.raw_alpha_loc = nn.Parameter(torch.tensor(init_raw))
        self.raw_alpha_coarse = nn.Parameter(torch.tensor(init_raw))

        # C1: 3-term weighted FPS (alpha, beta, gamma) via raw_weights
        self.anchor_selector = PhysicsAwareAnchorSelector(
            init_lambda=0.3,
            init_weights=(2.0, 1.0, 1.0),  # geometry-heavy start
        )
        self.learnable_coarse = LearnableCoarseOperator(
            spatial_dim=self.spatial_dim, feat_dim=op_hd, hidden_dim=op_hd, k_coarse=self.k_coarse)
        self.dual_scale_corrector = DualScaleGNNCorrector(
            spatial_dim=self.spatial_dim, feature_dim=self.feature_dim,
            output_dim=self.output_dim, hidden_dim=cfg["residual_hidden_dim"],
            num_fine_layers=cfg["residual_num_layers"], num_coarse_layers=cfg["coarse_num_layers"],
            k_virtual_nodes=cfg["k_virtual_nodes"])

        # ── caches ───────────────────────────────────────────
        self._mesh_cache: Optional[Dict[str, Any]] = None     # static, mesh-only
        self._mesh_key = None
        self._residual_cache: Optional[torch.Tensor] = None   # [N] |PDE residual|
        self._grad_norm_cache: Optional[torch.Tensor] = None  # [N] ||grad u||
        self._step_counter: int = 0

        # ── anchor recording (for visualisation) ─────────────
        self.record_anchors: bool = False
        self.anchor_history: List[Dict[str, Any]] = []        # [{'t': int, 'idx': LongTensor[m]}]

    # ── learnable operator scales ─────────────────────────────
    @property
    def alpha_loc(self):
        return F.softplus(self.raw_alpha_loc)

    @property
    def alpha_coarse_op(self):
        return F.softplus(self.raw_alpha_coarse)

    # ── static (mesh-only) graph ──────────────────────────────
    def _build_static_graph(self, nodes, edges, L_physics, device):
        N = nodes.shape[0]
        fine_ei = build_bidirectional_edges(edges)
        fine_ea = build_edge_features(nodes, fine_ei)

        if isinstance(L_physics, dict) and L_physics.get("type") == "sparse":
            sp_ei = L_physics["edge_index"].to(device)
            sp_w = L_physics["edge_weights"].to(device)
        else:
            L_dense = L_physics if isinstance(L_physics, torch.Tensor) else torch.zeros(N, N, device=device)
            sp_mask = (L_dense != 0)
            sp_idx = sp_mask.nonzero(as_tuple=True)
            sp_ei = torch.stack(sp_idx, dim=0)
            sp_w = L_dense[sp_idx]

        Llocal_weights = _match_weights_to_edge_index(sp_ei, sp_w, fine_ei, N)
        return {"N": N, "fine_ei": fine_ei, "fine_ea": fine_ea, "Llocal_weights": Llocal_weights}

    # ── anchor count target (with configurable cap) ───────────
    def _m_target(self, N):
        return min(self.m_anchors, max(8, N // self.anchor_cap_ratio), self.anchor_cap_max)

    # ── select anchors using the CURRENT physics caches ───────
    def _select_anchors(self, nodes, device):
        m_target = self._m_target(nodes.shape[0])
        return self.anchor_selector(
            nodes, m_target,
            residual=self._residual_cache,
            grad_norm=self._grad_norm_cache,
            use_physics_anchor=self.use_physics_anchor,
        )

    # ── build R / P / coarse graph for a given anchor set ─────
    def _build_anchor_graph(self, nodes, anchor_idx):
        anchor_coords = nodes[anchor_idx]
        m = anchor_idx.shape[0]
        R, P = build_restriction_prolongation(nodes, anchor_idx, q=self.q_local)
        k_c = min(self.k_coarse, m - 1)
        coarse_ei, _ = build_knn_graph(anchor_coords, k_c)
        coarse_ea = build_coarse_edge_attr(anchor_coords, coarse_ei)
        return {
            "m": m, "anchor_idx": anchor_idx, "anchor_coords": anchor_coords,
            "R": R, "P": P, "coarse_ei": coarse_ei, "coarse_ea": coarse_ea,
        }

    # ── effective Laplacian matvec (fine local + coarse) ──────
    def _Leff_matvec(self, v, gc, anchor_feats):
        Lv_loc = _sparse_Lv(v, gc["Llocal_weights"], gc["fine_ei"])
        R, P = gc["R"], gc["P"]
        coarse_ei = gc["coarse_ei"]
        anchor_coords = gc["anchor_coords"]
        v_c = R @ v
        L_hat = self.learnable_coarse(
            anchor_coords, anchor_feats, coarse_ei,
            use_learned_coarse=self.use_learned_coarse)
        Lv_c = L_hat @ v_c
        Lv_coarse = P @ Lv_c
        return self.alpha_loc * Lv_loc + self.alpha_coarse_op * Lv_coarse

    # ── refresh residual + gradient caches from a field u0 ────
    def _update_physics_caches(self, u0, gc, anchor_feats, f_term, faces, nodes, device, N):
        with torch.no_grad():
            Lu = self._Leff_matvec(u0, gc, anchor_feats.detach())
            self._residual_cache = (Lu + f_term).abs().detach()
            if faces is not None:
                self._grad_norm_cache = compute_gradient_norm(nodes, u0, faces).detach()
            else:
                src_e, dst_e = gc["fine_ei"]
                du = (u0[dst_e] - u0[src_e]).abs()
                g = torch.zeros(N, device=device, dtype=u0.dtype)
                g.scatter_add_(0, src_e, du)
                cnt = torch.zeros(N, device=device, dtype=u0.dtype)
                cnt.scatter_add_(0, src_e, torch.ones_like(du))
                self._grad_norm_cache = (g / cnt.clamp(min=1)).detach()

    def _record(self, t, gc):
        if self.record_anchors:
            self.anchor_history.append({"t": int(t), "idx": gc["anchor_idx"].detach().cpu().clone()})

    # ── forward ───────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any],
                use_physics_anchor=None, use_learned_coarse=None,
                use_dual_scale_gnn=None, use_virtual_nodes=None,
                record_anchors: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        _pa = use_physics_anchor if use_physics_anchor is not None else self.use_physics_anchor
        _lc = use_learned_coarse if use_learned_coarse is not None else self.use_learned_coarse
        _ds = use_dual_scale_gnn if use_dual_scale_gnn is not None else self.use_dual_scale_gnn
        _vn = use_virtual_nodes if use_virtual_nodes is not None else self.use_virtual_nodes
        if record_anchors is not None:
            self.record_anchors = bool(record_anchors)

        nodes = batch["nodes"]
        edges = batch["edges"]
        faces = batch.get("faces")
        node_type = batch.get("node_type",
                              torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))
        bnd_info = batch.get("boundary_info", {})
        L_physics = batch.get("L_physics", None)
        src_terms = batch["source_terms"]
        time_pts = batch["time_points"]
        u_init = batch["initial_conditions"]

        B, T, N, C = src_terms.shape
        device = nodes.device
        dt = float(time_pts[1] - time_pts[0]) if T > 1 else 0.0

        # ── static graph (mesh-only cache) ───────────────────
        mesh_key = (N, str(device))
        if self._mesh_cache is None or self._mesh_key != mesh_key:
            if L_physics is None:
                try:
                    from physics import build_operator
                    L_physics = build_operator(
                        nodes=nodes, edges=edges, faces=faces,
                        node_volumes=batch.get("node_volumes"),
                        operator_type=self.cfg.get("operator_type", "laplace"))
                except Exception:
                    L_physics = torch.zeros(N, N, device=device)
            self._mesh_cache = self._build_static_graph(nodes, edges, L_physics, device)
            self._mesh_key = mesh_key
        gc = dict(self._mesh_cache)  # shallow copy; dynamic keys added below

        _nv = batch.get("node_volumes")
        if _nv is None:
            _nv = torch.ones(N, device=device, dtype=nodes.dtype)

        # node features (with grad) — reused to index anchor_feats after re-selection
        node_feats_all = self.fine_encoder(nodes, gc["fine_ei"], gc["fine_ea"], _nv, node_type)

        # reset physics caches per rollout so anchors reflect THIS trajectory
        self._residual_cache = None
        self._grad_norm_cache = None
        if self.record_anchors:
            self.anchor_history = []

        # ── initial anchors (geometry first) ─────────────────
        anchor_idx = self._select_anchors(nodes, device)
        gc.update(self._build_anchor_graph(nodes, anchor_idx))
        anchor_feats = node_feats_all[gc["anchor_idx"]]

        # ── physics-aware initial RE-selection from u_init ───
        #    (this is what makes C1 actually do something)
        if _pa:
            self._update_physics_caches(u_init[0, :, 0], gc, anchor_feats,
                                        src_terms[0, 0, :, 0], faces, nodes, device, N)
            anchor_idx = self._select_anchors(nodes, device)
            gc.update(self._build_anchor_graph(nodes, anchor_idx))
            anchor_feats = node_feats_all[gc["anchor_idx"]]
        self._record(0, gc)

        precond_inv = _build_jacobi_precond(gc["Llocal_weights"], gc["fine_ei"], N, dt)

        u_hist = torch.zeros(B, T, N, C, device=device)
        u_curr = u_init
        u_hist[:, 0] = u_curr
        cg_warm_start = None

        for t in range(T - 1):
            f_cur = src_terms[:, t]
            f_next = src_terms[:, t + 1]

            # ── DYNAMIC anchor update (the core fix) ──────────
            if _pa and self.dynamic_anchors and t > 0 and (t % self.res_upd_freq == 0):
                self._update_physics_caches(u_curr[0, :, 0], gc, anchor_feats,
                                            f_cur[0, :, 0], faces, nodes, device, N)
                anchor_idx = self._select_anchors(nodes, device)
                gc.update(self._build_anchor_graph(nodes, anchor_idx))
                anchor_feats = node_feats_all[gc["anchor_idx"]]
                self._record(t, gc)
            self._step_counter += 1

            # ── physics path: CG implicit solve ───────────────
            u_phys_next = torch.zeros_like(u_curr)
            for b in range(B):
                u_b = u_curr[b]

                def Bop_mv(v):
                    return v + (dt / 2.0) * self._Leff_matvec(v, gc, anchor_feats)

                rhs_b = (Bop_mv(u_b[:, 0]) +
                         (dt / 2.0) * (f_cur[b, :, 0] + f_next[b, :, 0])).unsqueeze(-1)

                def A_mv_scalar(v):
                    return v - (dt / 2.0) * self._Leff_matvec(v, gc, anchor_feats)

                x0 = cg_warm_start[b][:, 0] if cg_warm_start is not None else u_b[:, 0]
                x_sol = _precond_cg_solve(A_mv_scalar, rhs_b[:, 0],
                                          precond_inv=precond_inv,
                                          max_iter=self.cg_max_iter,
                                          tol=self.cg_tol, x0=x0)
                u_phys_next[b] = x_sol.unsqueeze(-1)
            cg_warm_start = u_phys_next.detach()

            # ── C3: dual-scale GNN correction ─────────────────
            u_corr = torch.zeros_like(u_curr)
            for b in range(B):
                u_corr[b] = self.dual_scale_corrector(
                    u=u_curr[b], nodes=nodes,
                    edge_index=gc["fine_ei"], edge_attr=gc["fine_ea"],
                    node_type=node_type,
                    anchor_idx=gc["anchor_idx"], anchor_coords=gc["anchor_coords"],
                    R=gc["R"], P=gc["P"],
                    coarse_edge_index=gc["coarse_ei"],
                    coarse_edge_attr=gc["coarse_ea"],
                    use_dual_scale=_ds, use_virtual_nodes=_vn)

            u_next = u_phys_next + u_corr
            if bnd_info:
                u_next = _apply_bcs(u_next, bnd_info)
            u_hist[:, t + 1] = u_next
            u_curr = u_next.detach()

        return {"u_final": u_hist}

    # ── utilities ─────────────────────────────────────────────
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def ablation_summary(self):
        anchor_w = self.anchor_selector.weight_summary()
        lines = [
            "=== PhysHGNet Ablation Config ===",
            f"  C1 Physics Anchor : {'ON' if self.use_physics_anchor else 'OFF'}",
            f"  Dynamic Anchors   : {'ON' if self.dynamic_anchors else 'OFF'} "
            f"(update every {self.res_upd_freq} steps)",
            f"  Anchor weights    : {anchor_w}",
            f"  C2 Learned Coarse : {'ON' if self.use_learned_coarse else 'OFF'}",
            f"  C3 Dual-Scale GNN : {'ON' if self.use_dual_scale_gnn else 'OFF'}",
            f"  C3 Virtual Nodes  : {'ON' if self.use_virtual_nodes else 'OFF'}",
            f"  Total Parameters  : {self.num_parameters():,}",
        ]
        return "\n".join(lines)
