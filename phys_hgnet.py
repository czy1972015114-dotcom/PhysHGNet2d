"""
phys_hgnet.py — PhysHGNet: Physics-aware Hierarchical Graph Neural Operator.
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
    "dynamic_anchors": True,
    "anchor_cap_ratio": 16,
    "anchor_cap_max": 256,
    "use_mg_precond": True,
    # ── C1 辅助损失权重 ──────────────────────────────────────
    "anchor_temp_loss_weight": 0.1,   # L_anchor 在总 loss 中的权重
}


def _sparse_Lv(v, Lp_w, fine_ei):
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


# ══════════════════════════════════════════════════════════════════════════════
# 温度优先加权 FPS（推理层：保证锚点物理位置在热区）
# ══════════════════════════════════════════════════════════════════════════════
def _temperature_weighted_fps(nodes: torch.Tensor,
                               T_pos: torch.Tensor,
                               m: int,
                               hot_pct: float = 0.04) -> torch.Tensor:
    """
    温度优先加权 FPS — 时间自适应版本。

    候选池 = 前 hot_pct*N 个最热节点（固定比例，不受热量扩散影响）。
    FPS score = dist^0.1 * T_norm^8：温度绝对主导，距离极弱。

    hot_pct=0.04：N=2000 时取 top-80，严格锁定在黄色热斑内。
    """
    N = nodes.shape[0]
    k_cand = max(m + 1, int(N * hot_pct))
    k_cand = min(k_cand, N)

    _, cand_global = T_pos.topk(k_cand)
    cand_coords = nodes[cand_global]
    cand_T      = T_pos[cand_global]

    # 候选池内 z-score + minmax → [0,1]，保持相对温差在各时刻可比
    T_mean = cand_T.mean()
    T_std  = cand_T.std().clamp(min=1e-6)
    T_z    = (cand_T - T_mean) / T_std
    T_z_min, T_z_max = T_z.min(), T_z.max()
    T_norm = (T_z - T_z_min) / (T_z_max - T_z_min + 1e-8)
    cand_w = (T_norm ** 8).clamp(min=1e-6)   # 八次方：峰值节点权重 >> 次热节点

    selected_local = []
    start = cand_T.argmax().item()
    selected_local.append(start)
    min_dist = torch.full((k_cand,), float('inf'),
                          device=nodes.device, dtype=nodes.dtype)

    for _ in range(m - 1):
        last = cand_coords[selected_local[-1]]
        d = (cand_coords - last).norm(dim=-1).clamp(min=1e-8)
        min_dist = torch.minimum(min_dist, d)
        score = (min_dist ** 0.1) * (cand_w ** 8)   # 距离^0.1：几乎只看温度
        for s in selected_local:
            score[s] = -1.0
        selected_local.append(score.argmax().item())

    local_idx = torch.tensor(selected_local, device=nodes.device, dtype=torch.long)
    return cand_global[local_idx]


# ══════════════════════════════════════════════════════════════════════════════
# C1：可微物理感知锚点池化
#
# 网络设计层面的温度引导：
#   推理层（FPS）：hot_pct=0.04 + T^8，锚点物理位置严格在热区
#   训练层（辅助损失）：L_anchor = -Σ_j mean_i(P[i,j]·T̃[i])
#     → 直接训练 encoder + anchor_protos，让每个锚点的"管辖域"覆盖最热节点
#     → 梯度路径：L_anchor → P → softmax(spatial+feat+T_bonus) → encoder/protos
#
# 辅助损失的物理意义：
#   P[i,j] = 节点 i 属于锚点 j 的软分配权重
#   Σ_i P[i,j]·T̃[i] = 锚点 j 覆盖区域的平均归一化温度
#   最大化此值 = 训练网络让每个锚点尽可能覆盖高温节点
# ══════════════════════════════════════════════════════════════════════════════
class DifferentiableAnchorPooling(nn.Module):

    def __init__(self, spatial_dim: int = 2, feat_dim: int = 1,
                 hidden: int = 64, m_anchors: int = 64):
        super().__init__()
        node_in = spatial_dim + feat_dim + 3

        # 物理特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(node_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # m 个可学习锚点原型向量
        self.anchor_protos = nn.Parameter(
            torch.randn(m_anchors, hidden) / (hidden ** 0.5))

        self.log_temp  = nn.Parameter(torch.tensor(0.0))
        self.gamma_temp = nn.Parameter(torch.tensor(0.5413))
        self.log_sigma = nn.Parameter(torch.tensor(-3.0))

        # 辅助损失：上一次 forward 的 L_anchor（供训练脚本累加到总 loss）
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    @property
    def temp(self):
        return self.log_temp.exp().clamp(min=0.05, max=5.0)

    def weight_summary(self):
        gamma = F.softplus(self.gamma_temp).item()
        sigma = self.log_sigma.exp().item()
        return f"temp={self.temp.item():.3f}  gamma_T={gamma:.3f}  sigma={sigma:.3f}"

    def forward(self, nodes: torch.Tensor, u_curr: torch.Tensor,
                node_type: torch.Tensor, use_physics: bool = True):
        """
        返回: P, R, anchor_coords, anchor_feats, S
        副作用: 更新 self.last_aux_loss（训练时加入总损失）
        """
        N, d = nodes.shape
        m    = self.anchor_protos.shape[0]
        H    = self.anchor_protos.shape[1]
        device = nodes.device
        T = u_curr.squeeze(-1)   # [N]

        if use_physics:
            # ── 推理层：温度加权 FPS 决定锚点物理位置 ─────────────────────
            with torch.no_grad():
                T_min = T.min()
                T_pos = (T - T_min).clamp(min=0.0)
                if T_pos.max() < 1e-8:
                    seed_idx = torch.randperm(N, device=device)[:m]
                else:
                    seed_idx = _temperature_weighted_fps(
                        nodes.detach(), T_pos, m, hot_pct=0.04)

            anchor_coords = nodes[seed_idx]   # [m, d]，直接在热区节点上

            # ── 特征编码 ───────────────────────────────────────────────────
            nt   = F.one_hot(node_type.long(), 3).float()
            x_in = torch.cat([nodes, u_curr, nt], dim=-1)
            phi  = self.encoder(x_in)          # [N, H]

            # ── 软分配（训练层：encoder+protos 通过辅助 loss 被引导）──────
            sigma   = self.log_sigma.exp().clamp(min=0.005, max=0.5)
            dists   = torch.cdist(nodes, anchor_coords.detach())
            spatial = -dists / sigma

            feat    = phi @ self.anchor_protos.T / self.temp

            T_mean  = T.mean()
            T_std   = T.std().clamp(min=1e-4)
            T_norm  = (T - T_mean) / T_std
            gamma   = F.softplus(self.gamma_temp)
            T_bonus = gamma * T_norm.unsqueeze(-1)

            logits  = spatial + feat + T_bonus
            S       = torch.softmax(logits, dim=-1)   # [N, m]

            # 温度加权列归一化
            T_weight = F.softplus(T_norm + 1.0).unsqueeze(-1)
            S_temp   = S * T_weight
            col_sums = S_temp.sum(0, keepdim=True).clamp(min=1e-8)
            P        = S_temp / col_sums              # [N, m]

            # ── 训练层辅助损失：L_anchor ────────────────────────────────────
            # 每个锚点 j 的归一化温度覆盖 = Σ_i P[i,j] * T̃[i]
            # 其中 T̃ = (T - T_min) / (T_max - T_min)，映射到 [0,1]
            # 最大化此值 → encoder+protos 学习让锚点覆盖热节点
            #
            # 梯度路径：L_anchor → P → softmax → (feat项) → encoder, anchor_protos
            #           同时通过 T_bonus → gamma_temp
            if self.training:
                T_max  = T.max().clamp(min=T_min + 1e-8)
                T_norm01 = (T - T_min) / (T_max - T_min)    # [N], 归一化到[0,1]
                # anchor_temp_coverage[j] = 锚点j覆盖域的平均归一化温度
                anchor_temp_coverage = P.T @ T_norm01        # [m]
                # 辅助损失 = 负覆盖温度（最小化 = 最大化覆盖温度）
                # clamp防止梯度爆炸
                self.last_aux_loss = -anchor_temp_coverage.mean().clamp(min=-1.0, max=0.0)
            else:
                self.last_aux_loss = torch.tensor(0.0, device=device)

        else:
            # no_c1：均匀采样，无温度偏置
            with torch.no_grad():
                seed_idx = torch.randperm(N, device=device)[:m]
            anchor_coords = nodes[seed_idx]
            S        = torch.full((N, m), 1.0 / m, device=device)
            phi      = torch.zeros(N, H, device=device)
            col_sums = torch.full((1, m), float(N) / m, device=device)
            P        = S / col_sums
            self.last_aux_loss = torch.tensor(0.0, device=device)

        R = P.T
        anchor_feats = R @ phi

        return P, R, anchor_coords, anchor_feats, S


class PhysHGNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **config}
        self.cfg = cfg

        self.spatial_dim  = cfg["spatial_dim"]
        self.feature_dim  = cfg["feature_dim"]
        self.output_dim   = cfg["output_dim"]
        self.m_anchors    = cfg["m_anchors"]
        self.q_local      = cfg["q_local"]
        self.k_coarse     = cfg["k_coarse"]
        self.cg_max_iter  = cfg["cg_max_iter"]
        self.cg_tol       = cfg["cg_tol"]
        self.res_upd_freq = max(1, int(cfg["residual_update_freq"]))

        self.use_physics_anchor  = cfg["use_physics_anchor"]
        self.use_learned_coarse  = cfg["use_learned_coarse"]
        self.use_dual_scale_gnn  = cfg["use_dual_scale_gnn"]
        self.use_virtual_nodes   = cfg["use_virtual_nodes"]
        self.dynamic_anchors     = bool(cfg["dynamic_anchors"])
        self.anchor_cap_ratio    = max(1, int(cfg["anchor_cap_ratio"]))
        self.anchor_cap_max      = int(cfg["anchor_cap_max"])
        self.use_mg_precond      = bool(cfg.get("use_mg_precond", True))
        self.anchor_temp_loss_w  = float(cfg.get("anchor_temp_loss_weight", 0.1))

        op_hd = cfg["operator_hidden_dim"]
        op_nl = cfg["operator_num_layers"]
        self.fine_encoder = FineGraphEncoder(self.spatial_dim, op_hd, op_nl)

        init_raw = math.log(max(math.exp(0.001) - 1.0, 1e-10))
        self.raw_alpha_loc    = nn.Parameter(torch.tensor(init_raw))
        self.raw_alpha_coarse = nn.Parameter(torch.tensor(init_raw))

        self.anchor_pooling = DifferentiableAnchorPooling(
            spatial_dim=self.spatial_dim,
            feat_dim=self.feature_dim,
            hidden=op_hd,
            m_anchors=self.m_anchors,
        )
        self.learnable_coarse = LearnableCoarseOperator(
            spatial_dim=self.spatial_dim, feat_dim=op_hd,
            hidden_dim=op_hd, k_coarse=self.k_coarse)
        self.dual_scale_corrector = DualScaleGNNCorrector(
            spatial_dim=self.spatial_dim, feature_dim=self.feature_dim,
            output_dim=self.output_dim, hidden_dim=cfg["residual_hidden_dim"],
            num_fine_layers=cfg["residual_num_layers"],
            num_coarse_layers=cfg["coarse_num_layers"],
            k_virtual_nodes=cfg["k_virtual_nodes"])

        self._mesh_cache: Optional[Dict[str, Any]] = None
        self._mesh_key   = None
        self._residual_cache: Optional[torch.Tensor] = None
        self._grad_norm_cache: Optional[torch.Tensor] = None
        self._step_counter: int = 0

        self.record_anchors: bool = False
        self.anchor_history: List[Dict[str, Any]] = []
        self.last_cg_iters: List[int] = []

        # 累积辅助损失（供训练脚本使用）
        self.last_anchor_aux_loss: torch.Tensor = torch.tensor(0.0)

    @property
    def alpha_loc(self):
        return F.softplus(self.raw_alpha_loc)

    @property
    def alpha_coarse_op(self):
        return F.softplus(self.raw_alpha_coarse)

    def _build_static_graph(self, nodes, edges, L_physics, device):
        N = nodes.shape[0]
        fine_ei = build_bidirectional_edges(edges)
        fine_ea = build_edge_features(nodes, fine_ei)
        if isinstance(L_physics, dict) and L_physics.get("type") == "sparse":
            sp_ei = L_physics["edge_index"].to(device)
            sp_w  = L_physics["edge_weights"].to(device)
        else:
            L_dense = L_physics if isinstance(L_physics, torch.Tensor) else torch.zeros(N, N, device=device)
            sp_mask = (L_dense != 0)
            sp_idx  = sp_mask.nonzero(as_tuple=True)
            sp_ei   = torch.stack(sp_idx, dim=0)
            sp_w    = L_dense[sp_idx]
        Llocal_weights = _match_weights_to_edge_index(sp_ei, sp_w, fine_ei, N)
        return {"N": N, "fine_ei": fine_ei, "fine_ea": fine_ea,
                "Llocal_weights": Llocal_weights}

    def _m_target(self, N):
        return min(self.m_anchors, max(8, N // self.anchor_cap_ratio), self.anchor_cap_max)

    def _select_anchors(self, nodes, device):
        raise RuntimeError('_select_anchors is deprecated; use anchor_pooling instead')

    def _build_soft_anchor_graph(self, anchor_coords, P, R, anchor_feats, S):
        m   = anchor_coords.shape[0]
        k_c = min(self.k_coarse, m - 1)
        coarse_ei, _ = build_knn_graph(anchor_coords.detach(), k_c)
        coarse_ea    = build_coarse_edge_attr(anchor_coords.detach(), coarse_ei)
        anchor_repr_idx = S.detach().argmax(dim=0)   # [m]
        N = S.shape[0]
        P_sparse = torch.zeros(N, m, device=S.device, dtype=S.dtype)
        P_sparse[anchor_repr_idx, torch.arange(m, device=S.device)] = 1.0
        R_sparse = P_sparse.T
        return {
            "m": m,
            "anchor_idx":      anchor_repr_idx,
            "anchor_hard_idx": anchor_repr_idx,
            "anchor_coords":   anchor_coords,
            "anchor_feats":    anchor_feats,
            "P":               P,
            "R":               R,
            "P_mg":            P_sparse,
            "R_mg":            R_sparse,
            "coarse_ei":       coarse_ei,
            "coarse_ea":       coarse_ea,
        }

    def _build_anchor_graph(self, nodes, anchor_idx):
        anchor_coords = nodes[anchor_idx]
        m = anchor_idx.shape[0]
        R, P = build_restriction_prolongation(nodes, anchor_idx, q=self.q_local)
        k_c = min(self.k_coarse, m - 1)
        coarse_ei, _ = build_knn_graph(anchor_coords, k_c)
        coarse_ea = build_coarse_edge_attr(anchor_coords, coarse_ei)
        return {"m": m, "anchor_idx": anchor_idx, "anchor_coords": anchor_coords,
                "R": R, "P": P, "coarse_ei": coarse_ei, "coarse_ea": coarse_ea}

    def _Leff_matvec(self, v, gc, anchor_feats):
        Lv_loc = _sparse_Lv(v, gc["Llocal_weights"], gc["fine_ei"])
        R, P   = gc["R"], gc["P"]
        v_c    = R @ v
        L_hat  = self.learnable_coarse(
            gc["anchor_coords"], anchor_feats, gc["coarse_ei"],
            use_learned_coarse=self.use_learned_coarse)
        Lv_c      = L_hat @ v_c
        Lv_coarse = P @ Lv_c
        return self.alpha_loc * Lv_loc + self.alpha_coarse_op * Lv_coarse

    def _build_mg_precond(self, gc, anchor_feats, jac_inv, dt):
        with torch.no_grad():
            L_hat = self.learnable_coarse(
                gc["anchor_coords"], anchor_feats.detach(), gc["coarse_ei"],
                use_learned_coarse=self.use_learned_coarse).detach()
            m   = L_hat.shape[0]
            I_m = torch.eye(m, device=L_hat.device, dtype=L_hat.dtype)
            alpha = float(self.alpha_coarse_op.detach().clamp(min=1e-6))
            A_c = I_m - (dt / 2.0) * alpha * L_hat
            A_c = (A_c + A_c.T) / 2.0 + 1e-5 * I_m
            try:
                lu, piv = torch.linalg.lu_factor(A_c)
                lu_ok = True
            except Exception:
                lu_ok = False
            P_norm = gc.get("P_mg", gc["P"])
            R_sym  = P_norm.T

        def M_inv(r):
            z = jac_inv * r
            if lu_ok:
                rc = R_sym @ r
                xc = torch.linalg.lu_solve(lu, piv, rc.unsqueeze(-1)).squeeze(-1)
                z  = z + P_norm @ xc
            return z
        return M_inv

    def _pcg(self, A_mv, b, M_inv, max_iter, tol, x0=None):
        x  = x0.clone() if x0 is not None else torch.zeros_like(b)
        r  = b - A_mv(x)
        bnorm = b.norm().clamp(min=1e-12)
        z  = M_inv(r)
        p  = z.clone()
        rz = (r * z).sum()
        iters = 0
        for k in range(max_iter):
            iters = k + 1
            Ap  = A_mv(p)
            pAp = (p * Ap).sum().clamp(min=1e-20)
            alpha = rz / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            if (r.norm() / bnorm) < tol:
                break
            z      = M_inv(r)
            rz_new = (r * z).sum()
            beta   = rz_new / rz.clamp(min=1e-20)
            p  = z + beta * p
            rz = rz_new
        return x, iters

    def _update_physics_caches(self, u0, gc, anchor_feats, f_term, faces, nodes, device, N):
        with torch.no_grad():
            Lu = self._Leff_matvec(u0, gc, anchor_feats.detach())
            self._residual_cache = (Lu + f_term).abs().detach()
            if faces is not None:
                self._grad_norm_cache = compute_gradient_norm(nodes, u0, faces).detach()
            else:
                src_e, dst_e = gc["fine_ei"]
                du  = (u0[dst_e] - u0[src_e]).abs()
                g   = torch.zeros(N, device=device, dtype=u0.dtype)
                g.scatter_add_(0, src_e, du)
                cnt = torch.zeros(N, device=device, dtype=u0.dtype)
                cnt.scatter_add_(0, src_e, torch.ones_like(du))
                self._grad_norm_cache = (g / cnt.clamp(min=1)).detach()

    def _record(self, t, gc):
        if self.record_anchors:
            self.anchor_history.append(
                {"t": int(t), "idx": gc["anchor_idx"].detach().cpu().clone()})

    def anchor_aux_loss(self) -> torch.Tensor:
        """供训练脚本调用：返回 anchor 温度辅助损失（已乘权重）"""
        return self.anchor_temp_loss_w * self.last_anchor_aux_loss

    def forward(self, batch: Dict[str, Any],
                use_physics_anchor=None, use_learned_coarse=None,
                use_dual_scale_gnn=None, use_virtual_nodes=None,
                record_anchors: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        _pa = use_physics_anchor  if use_physics_anchor  is not None else self.use_physics_anchor
        _lc = use_learned_coarse  if use_learned_coarse  is not None else self.use_learned_coarse
        _ds = use_dual_scale_gnn  if use_dual_scale_gnn  is not None else self.use_dual_scale_gnn
        _vn = use_virtual_nodes   if use_virtual_nodes   is not None else self.use_virtual_nodes
        if record_anchors is not None:
            self.record_anchors = bool(record_anchors)

        nodes     = batch["nodes"]
        edges     = batch["edges"]
        faces     = batch.get("faces")
        node_type = batch.get("node_type",
            torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))
        bnd_info  = batch.get("boundary_info", {})
        L_physics = batch.get("L_physics", None)
        src_terms = batch["source_terms"]
        time_pts  = batch["time_points"]
        u_init    = batch["initial_conditions"]

        B, T, N, C = src_terms.shape
        device = nodes.device
        dt = float(time_pts[1] - time_pts[0]) if T > 1 else 0.0

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
            self._mesh_key   = mesh_key
        gc = dict(self._mesh_cache)

        _nv = batch.get("node_volumes")
        if _nv is None:
            _nv = torch.ones(N, device=device, dtype=nodes.dtype)

        node_feats_all = self.fine_encoder(nodes, gc["fine_ei"], gc["fine_ea"], _nv, node_type)

        self._residual_cache  = None
        self._grad_norm_cache = None
        if self.record_anchors:
            self.anchor_history = []

        # 重置辅助损失累加器
        aux_loss_accum = torch.tensor(0.0, device=device)
        aux_loss_count = 0

        # ── C1: 初始帧锚点（使用 u_init 第一条轨迹）──────────────────────
        u0_field = u_init[0, :, 0].unsqueeze(-1)
        P0, R0, ac0, af0, S0 = self.anchor_pooling(
            nodes, u0_field, node_type, use_physics=_pa)
        if self.training:
            aux_loss_accum = aux_loss_accum + self.anchor_pooling.last_aux_loss
            aux_loss_count += 1
        gc.update(self._build_soft_anchor_graph(ac0, P0, R0, af0, S0))
        anchor_feats = gc["anchor_feats"]
        self._record(0, gc)

        precond_inv = _build_jacobi_precond(gc["Llocal_weights"], gc["fine_ei"], N, dt)

        u_hist = torch.zeros(B, T, N, C, device=device)
        u_curr = u_init
        u_hist[:, 0] = u_curr
        cg_warm_start = None
        self.last_cg_iters = []

        for t in range(T - 1):
            f_cur  = src_terms[:, t]
            f_next = src_terms[:, t + 1]

            if self.dynamic_anchors and t > 0 and (t % self.res_upd_freq == 0):
                u_t_field = u_curr[0, :, 0].unsqueeze(-1)
                Pt, Rt, act, aft, St = self.anchor_pooling(
                    nodes, u_t_field, node_type, use_physics=_pa)
                if self.training:
                    aux_loss_accum = aux_loss_accum + self.anchor_pooling.last_aux_loss
                    aux_loss_count += 1
                gc.update(self._build_soft_anchor_graph(act, Pt, Rt, aft, St))
                anchor_feats = gc["anchor_feats"]
                self._record(t, gc)
                precond_inv  = _build_jacobi_precond(
                    gc["Llocal_weights"], gc["fine_ei"], N, dt)
            self._step_counter += 1

            if self.use_mg_precond:
                M_inv = self._build_mg_precond(gc, anchor_feats, precond_inv, dt)
            else:
                M_inv = (lambda r: precond_inv * r)

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
                x_sol, n_it = self._pcg(A_mv_scalar, rhs_b[:, 0],
                                        M_inv=M_inv,
                                        max_iter=self.cg_max_iter,
                                        tol=self.cg_tol, x0=x0)
                self.last_cg_iters.append(int(n_it))
                u_phys_next[b] = x_sol.unsqueeze(-1)
            cg_warm_start = u_phys_next.detach()

            u_corr = torch.zeros_like(u_curr)
            for b in range(B):
                u_corr[b] = self.dual_scale_corrector(
                    u=u_curr[b], nodes=nodes,
                    edge_index=gc["fine_ei"], edge_attr=gc["fine_ea"],
                    node_type=node_type,
                    anchor_idx=gc["anchor_hard_idx"],
                    anchor_coords=gc["anchor_coords"],
                    R=gc["R"], P=gc["P"],
                    coarse_edge_index=gc["coarse_ei"],
                    coarse_edge_attr=gc["coarse_ea"],
                    use_dual_scale=_ds, use_virtual_nodes=_vn)

            u_next = u_phys_next + u_corr
            if bnd_info:
                u_next = _apply_bcs(u_next, bnd_info)
            u_hist[:, t + 1] = u_next
            u_curr = u_next.detach()

        # 保存平均辅助损失
        if aux_loss_count > 0:
            self.last_anchor_aux_loss = aux_loss_accum / aux_loss_count
        else:
            self.last_anchor_aux_loss = torch.tensor(0.0, device=device)

        return {"u_final": u_hist}

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def avg_cg_iters(self):
        return (sum(self.last_cg_iters) / len(self.last_cg_iters)
                if self.last_cg_iters else 0.0)

    def ablation_summary(self):
        anchor_w = self.anchor_pooling.weight_summary()
        lines = [
            "=== PhysHGNet Ablation Config ===",
            f"  C1 Physics Anchor  : {'ON' if self.use_physics_anchor else 'OFF'}",
            f"  Dynamic Anchors    : {'ON' if self.dynamic_anchors else 'OFF'} "
            f"(update every {self.res_upd_freq} steps)",
            f"  MG Preconditioner  : {'ON' if self.use_mg_precond else 'OFF'}",
            f"  Anchor temp loss w : {self.anchor_temp_loss_w}",
            f"  Anchor weights     : {anchor_w}",
            f"  C2 Learned Coarse  : {'ON' if self.use_learned_coarse else 'OFF'}",
            f"  C3 Dual-Scale GNN  : {'ON' if self.use_dual_scale_gnn else 'OFF'}",
            f"  C3 Virtual Nodes   : {'ON' if self.use_virtual_nodes else 'OFF'}",
            f"  Total Parameters   : {self.num_parameters():,}",
        ]
        return "\n".join(lines)
