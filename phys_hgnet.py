"""
phys_hgnet.py — PhysHGNet: Physics-aware Hierarchical Graph Neural Operator.

Extends StructuredDGNet with three innovations:
    C1 - PhysicsAwareAnchorSelector (residual + gradient-weighted FPS)
    C2 - LearnableCoarseOperator    (GSL coarse operator)
    C3 - DualScaleGNNCorrector      (dual-scale GNN with virtual nodes)

All innovations are individually toggleable via config for ablation studies.
Interface: forward(batch) -> {'u_final': (B, T, N, C)}
Batch format is identical to DGNet (from dataset.py).

────────────────────────────────────────────────────────────────────────
修复 / 改动说明（相对原始仓库版本）
────────────────────────────────────────────────────────────────────────
[FIX-1] 原文件 `from structured_dgnet import ImplicitCGSolve, _precond_cg_solve,
        ResidualSolver` —— structured_dgnet.py 里并不存在 `_precond_cg_solve`
        与 `ResidualSolver`，会直接 ImportError。这里：
          · 删除该错误导入；
          · 在本文件内置一个可微分（尊重外部 grad context）的预条件共轭梯度
            求解器 `_precond_cg_solve`，签名与原调用处一致；
          · `ImplicitCGSolve` / `ResidualSolver` 在本文件中本就未被使用，移除。

[FIX-2] 实时更新粗网格（创新点②）。原版把 `_build_graph`（含锚点选择）用
        `(N, device, use_physics_anchor)` 作为缓存键，只在首次 forward 选一次
        锚点后永久冻结，残差/梯度缓存虽然在更新却再也不会进入锚点选择 ——
        “实时更新粗网格”实际上没有发生。现在：
          · 细网格（边、边特征、L 权重、Jacobi 预条件）只随网格构建一次并缓存；
          · 锚点 + R/P + 粗图每隔 `coarse_update_freq` 个时间步用最新的
            残差 + 梯度信号重新选择，真正实现锚点随温度场实时迁移；
          · 第一帧即用 ‖∇u_init‖ 作为物理信号选锚点（无需先有锚点）。

[FIX-3] 把可学习粗算子 L_hat 的计算从 CG 每次迭代中提到“每个时间步算一次”。
        L_hat 在一次线性求解内是常量，原写法在 50 次 CG 迭代里反复重算
        MLP，纯属浪费且会撑大自动微分图。提取后数学完全等价，显著加速训练
        与推理（支撑“快速推理”这一创新点）。

[FIX-4] 放宽锚点数上限。原 `m_target = min(m_anchors, max(8, N//16), 256)`，
        在 N=4000 时把锚点硬截断到 250，导致锚点 scaling 实验（64/128/256/512）
        中 256、512 退化为 250。改为 `min(m_anchors, max(8, N//2))`。

[FIX-5] `forward(..., return_anchor_history=True)` 返回每个时间步生效的锚点
        索引列表，供 2.6 锚点可视化使用。
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
    "coarse_update_freq": 5,   # [FIX-2] 每隔多少个时间步重选一次锚点（=1 即每步更新）
    "operator_type": "laplace",
}


# ══════════════════════════════════════════════════════════════
# 内置：可微分预条件共轭梯度求解器  [FIX-1]
# ══════════════════════════════════════════════════════════════
def _precond_cg_solve(A_mv, b, precond_inv, max_iter=50, tol=1e-6, x0=None):
    """
    解 SPD 系统 A x = b 的（Jacobi）预条件共轭梯度。

    · 不加 @torch.no_grad，因此会“尊重外部的梯度上下文”：训练时（enable_grad）
      梯度可经由 A_mv 中的 learnable_coarse / fine_encoder / α 反传，C2 才能学习；
      推理时（外层 no_grad）则纯前向，速度快。
    · 支持热启动 x0（warm start），配合时间步进显著减少迭代数 —— “快速推理”。

    Args:
        A_mv:        callable, v[N] -> A v[N]
        b:           [N]
        precond_inv: [N]  Jacobi 预条件子（对角线倒数，正定）
        x0:          [N] or None  热启动初值
    Returns:
        x: [N]
        n_iters: int  实际执行的 CG 迭代次数（用于统计 avg_cg_iters）
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_mv(x)

    z = precond_inv * r
    p = z.clone()
    rz = (r * z).sum()
    b_norm = b.norm().clamp(min=1e-30)

    n_iters = 0
    for _ in range(max_iter):
        n_iters += 1
        Ap = A_mv(p)
        pAp = (p * Ap).sum()
        # 防止除零 / 病态方向（保持可微）
        denom = torch.where(pAp.abs() < 1e-30, torch.full_like(pAp, 1e-30), pAp)
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if float(r.norm() / b_norm) < tol:
            break
        z = precond_inv * r
        rz_new = (r * z).sum()
        beta = rz_new / rz.clamp(min=1e-30)
        p = z + beta * p
        rz = rz_new
    return x, n_iters


# ══════════════════════════════════════════════════════════════
# 稀疏 Laplacian matvec / Jacobi 预条件 / 边权匹配 / 边界条件
# ══════════════════════════════════════════════════════════════
def _sparse_Lv(v, Lp_w, fine_ei):
    """Self-contained sparse Laplacian matvec: (L v)_i = Σ_j w_ij (v_j - v_i)."""
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


def _edge_grad_fallback(nodes, u0, fine_ei, N):
    """faces 缺失时用边差分近似 ‖∇u‖。"""
    src_e, dst_e = fine_ei
    du = (u0[dst_e] - u0[src_e]).abs()
    g = torch.zeros(N, device=nodes.device, dtype=u0.dtype)
    g.scatter_add_(0, src_e, du)
    cnt = torch.zeros(N, device=nodes.device, dtype=u0.dtype)
    cnt.scatter_add_(0, src_e, torch.ones_like(du))
    return g / cnt.clamp(min=1)


def _apply_bcs(u, boundary_info):
    if not isinstance(boundary_info, dict):
        return u
    di = boundary_info.get("dirichlet")
    if di is None:
        return u
    idx = di.get("indices")
    val = di.get("values")
    if idx is None or (hasattr(idx, "numel") and idx.numel() == 0):
        return u
    device = u.device
    idx = idx.to(device)
    u = u.clone()
    if val is not None:
        val = val.to(device)
        u[:, idx] = val.unsqueeze(0).unsqueeze(-1).expand(u.shape[0], -1, u.shape[-1])
    return u


class PhysHGNet(nn.Module):
    """
    Physics-aware Hierarchical Graph Neural Operator.

    相对 DGNet 的关键区别：
      1. CG 求解（非 LU）：O(N) 内存、可扩展性更好
      2. 分层多尺度结构（细图 + 粗图），粗图刻画长程传播
      3. C1: 物理残差 + 梯度加权的三项 FPS 锚点选择（并随时间实时更新）
      4. C2: 数据驱动的可学习粗算子
      5. C3: 含虚拟节点的双尺度 GNN（长程交互）
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
        self.res_upd_freq = cfg["residual_update_freq"]
        self.coarse_upd_freq = max(1, int(cfg["coarse_update_freq"]))

        self.use_physics_anchor = cfg["use_physics_anchor"]
        self.use_learned_coarse = cfg["use_learned_coarse"]
        self.use_dual_scale_gnn = cfg["use_dual_scale_gnn"]
        self.use_virtual_nodes = cfg["use_virtual_nodes"]

        # CG 迭代统计（供 train_physhgnet.py 的 avg_cg_iters() 使用）
        self._cg_iter_sum = 0
        self._cg_iter_count = 0

        op_hd = cfg["operator_hidden_dim"]
        op_nl = cfg["operator_num_layers"]
        self.fine_encoder = FineGraphEncoder(self.spatial_dim, op_hd, op_nl)

        init_raw = math.log(max(math.exp(0.001) - 1.0, 1e-10))
        self.raw_alpha_loc = nn.Parameter(torch.tensor(init_raw))
        self.raw_alpha_coarse = nn.Parameter(torch.tensor(init_raw))

        # C1: 三项加权 FPS（α,β,γ）
        self.anchor_selector = PhysicsAwareAnchorSelector(
            init_lambda=0.3, init_weights=(2.0, 1.0, 1.0))

        self.learnable_coarse = LearnableCoarseOperator(
            spatial_dim=self.spatial_dim, feat_dim=op_hd, hidden_dim=op_hd, k_coarse=self.k_coarse)

        self.dual_scale_corrector = DualScaleGNNCorrector(
            spatial_dim=self.spatial_dim, feature_dim=self.feature_dim,
            output_dim=self.output_dim, hidden_dim=cfg["residual_hidden_dim"],
            num_fine_layers=cfg["residual_num_layers"], num_coarse_layers=cfg["coarse_num_layers"],
            k_virtual_nodes=cfg["k_virtual_nodes"])

        # 细网格缓存（只随网格大小/设备改变）
        self._fine_cache: Optional[Dict[str, Any]] = None
        self._fine_cache_key = None

    @property
    def alpha_loc(self):
        return F.softplus(self.raw_alpha_loc)

    @property
    def alpha_coarse_op(self):
        return F.softplus(self.raw_alpha_coarse)

    # ── 细网格（一次性构建，按 N/device 缓存）─────────────────
    def _build_fine(self, nodes, edges, L_physics):
        N = nodes.shape[0]
        device = nodes.device
        fine_ei = build_bidirectional_edges(edges)
        fine_ea = build_edge_features(nodes, fine_ei)

        if isinstance(L_physics, dict) and L_physics.get("type") == "sparse":
            sp_ei, sp_w = L_physics["edge_index"].to(device), L_physics["edge_weights"].to(device)
        else:
            L_dense = L_physics if isinstance(L_physics, torch.Tensor) else torch.zeros(N, N, device=device)
            sp_mask = (L_dense != 0)
            sp_idx = sp_mask.nonzero(as_tuple=True)
            sp_ei = torch.stack(sp_idx, dim=0)
            sp_w = L_dense[sp_idx]

        Llocal_weights = _match_weights_to_edge_index(sp_ei, sp_w, fine_ei, N)
        # [FIX-4] 放宽锚点上限：仅受 N 约束
        m_target = min(self.m_anchors, max(8, N // 2))
        return {
            "N": N, "fine_ei": fine_ei, "fine_ea": fine_ea,
            "Llocal_weights": Llocal_weights, "m_target": m_target,
        }

    # ── 锚点选择 + R/P + 粗图（随物理信号刷新）  [FIX-2] ───────
    def _select_coarse(self, nodes, fine, node_volumes, node_type,
                       residual, grad_norm, use_physics_anchor):
        N = nodes.shape[0]
        device = nodes.device
        anchor_idx = self.anchor_selector(
            nodes, fine["m_target"],
            residual=residual, grad_norm=grad_norm,
            use_physics_anchor=use_physics_anchor,
        )
        anchor_coords = nodes[anchor_idx]
        m = anchor_idx.shape[0]
        R, P = build_restriction_prolongation(nodes, anchor_idx, q=self.q_local)
        k_c = min(self.k_coarse, max(1, m - 1))
        coarse_ei, _ = build_knn_graph(anchor_coords, k_c)
        coarse_ea = build_coarse_edge_attr(anchor_coords, coarse_ei)
        return {
            "m": m, "anchor_idx": anchor_idx, "anchor_coords": anchor_coords,
            "R": R, "P": P, "coarse_ei": coarse_ei, "coarse_ea": coarse_ea,
        }

    def _anchor_feats(self, nodes, fine, node_volumes, node_type, anchor_idx):
        feats = self.fine_encoder(nodes, fine["fine_ei"], fine["fine_ea"],
                                  node_volumes, node_type)
        return feats[anchor_idx]

    # ── 有效算子 L_eff = α_loc·L_loc + α_coarse·(P L_hat R) ────
    def _Leff_matvec(self, v, fine, coarse, L_hat):
        Lv_loc = _sparse_Lv(v, fine["Llocal_weights"], fine["fine_ei"])
        v_c = coarse["R"] @ v
        Lv_c = L_hat @ v_c
        Lv_coarse = coarse["P"] @ Lv_c
        return self.alpha_loc * Lv_loc + self.alpha_coarse_op * Lv_coarse

    # ── forward ───────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any],
                use_physics_anchor=None, use_learned_coarse=None,
                use_dual_scale_gnn=None, use_virtual_nodes=None,
                return_anchor_history: bool = False) -> Dict[str, torch.Tensor]:
        _pa = use_physics_anchor if use_physics_anchor is not None else self.use_physics_anchor
        _lc = use_learned_coarse if use_learned_coarse is not None else self.use_learned_coarse
        _ds = use_dual_scale_gnn if use_dual_scale_gnn is not None else self.use_dual_scale_gnn
        _vn = use_virtual_nodes if use_virtual_nodes is not None else self.use_virtual_nodes

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

        node_volumes = batch.get("node_volumes")
        if node_volumes is None:
            node_volumes = torch.ones(N, device=device, dtype=nodes.dtype)

        # ── 细网格（缓存）──────────────────────────────────
        fine_key = (N, str(device))
        if self._fine_cache is None or self._fine_cache_key != fine_key:
            if L_physics is None:
                try:
                    from physics import build_operator
                    L_physics = build_operator(
                        nodes=nodes, edges=edges, faces=faces,
                        node_volumes=node_volumes,
                        operator_type=self.cfg.get("operator_type", "laplace"))
                except Exception:
                    L_physics = torch.zeros(N, N, device=device)
            self._fine_cache = self._build_fine(nodes, edges, L_physics)
            self._fine_cache_key = fine_key
        fine = self._fine_cache

        precond_inv = _build_jacobi_precond(fine["Llocal_weights"], fine["fine_ei"], N, dt)

        # ── 初始物理信号（用 ‖∇u_init‖ 让第一帧锚点也物理感知）──
        with torch.no_grad():
            u0 = u_init[0, :, 0] if u_init.dim() == 3 else u_init[0, 0, :, 0]
            if _pa and faces is not None:
                grad_norm = compute_gradient_norm(nodes, u0, faces).detach()
            elif _pa:
                grad_norm = _edge_grad_fallback(nodes, u0, fine["fine_ei"], N).detach()
            else:
                grad_norm = None
        residual = None  # 第一帧尚无锚点，残差项为 0

        coarse = self._select_coarse(nodes, fine, node_volumes, node_type,
                                     residual=residual, grad_norm=grad_norm,
                                     use_physics_anchor=_pa)
        anchor_feats = self._anchor_feats(nodes, fine, node_volumes, node_type,
                                          coarse["anchor_idx"])

        u_hist = torch.zeros(B, T, N, C, device=device)
        u_curr = u_init
        u_hist[:, 0] = u_curr
        cg_warm_start = None
        anchor_history: List[torch.Tensor] = []
        self._cg_iter_sum = 0
        self._cg_iter_count = 0

        for t in range(T - 1):
            f_cur = src_terms[:, t]
            f_next = src_terms[:, t + 1]

            # ── 实时刷新锚点（残差 + 梯度信号）[FIX-2] ────────
            if _pa and t > 0 and (t % self.coarse_upd_freq == 0):
                with torch.no_grad():
                    u0 = u_curr[0, :, 0]
                    L_hat_det = self.learnable_coarse(
                        coarse["anchor_coords"], anchor_feats.detach(),
                        coarse["coarse_ei"], use_learned_coarse=_lc)
                    Lu = self._Leff_matvec(u0, fine, coarse, L_hat_det)
                    residual = (Lu + f_cur[0, :, 0]).abs().detach()
                    if faces is not None:
                        grad_norm = compute_gradient_norm(nodes, u0, faces).detach()
                    else:
                        grad_norm = _edge_grad_fallback(nodes, u0, fine["fine_ei"], N).detach()
                coarse = self._select_coarse(nodes, fine, node_volumes, node_type,
                                             residual=residual, grad_norm=grad_norm,
                                             use_physics_anchor=_pa)
                anchor_feats = self._anchor_feats(nodes, fine, node_volumes, node_type,
                                                  coarse["anchor_idx"])

            if return_anchor_history:
                anchor_history.append(coarse["anchor_idx"].detach().cpu())

            # ── L_hat 每个时间步只算一次  [FIX-3] ───────────
            L_hat = self.learnable_coarse(
                coarse["anchor_coords"], anchor_feats, coarse["coarse_ei"],
                use_learned_coarse=_lc)

            # ── 物理路径：CG 隐式求解 ───────────────────────
            u_phys_next = torch.zeros_like(u_curr)
            for b in range(B):
                u_b = u_curr[b]

                def Bop_mv(v):
                    return v + (dt / 2.0) * self._Leff_matvec(v, fine, coarse, L_hat)

                def A_mv_scalar(v):
                    return v - (dt / 2.0) * self._Leff_matvec(v, fine, coarse, L_hat)

                rhs_b = (Bop_mv(u_b[:, 0]) +
                         (dt / 2.0) * (f_cur[b, :, 0] + f_next[b, :, 0]))
                x0 = cg_warm_start[b][:, 0] if cg_warm_start is not None else u_b[:, 0]
                x_sol, cg_it = _precond_cg_solve(A_mv_scalar, rhs_b,
                                                 precond_inv=precond_inv,
                                                 max_iter=self.cg_max_iter,
                                                 tol=self.cg_tol, x0=x0)
                self._cg_iter_sum += cg_it
                self._cg_iter_count += 1
                u_phys_next[b] = x_sol.unsqueeze(-1)
            cg_warm_start = u_phys_next.detach()

            # ── C3: 双尺度 GNN 修正 ─────────────────────────
            u_corr = torch.zeros_like(u_curr)
            for b in range(B):
                u_corr[b] = self.dual_scale_corrector(
                    u=u_curr[b], nodes=nodes,
                    edge_index=fine["fine_ei"], edge_attr=fine["fine_ea"],
                    node_type=node_type,
                    anchor_idx=coarse["anchor_idx"], anchor_coords=coarse["anchor_coords"],
                    R=coarse["R"], P=coarse["P"],
                    coarse_edge_index=coarse["coarse_ei"],
                    coarse_edge_attr=coarse["coarse_ea"],
                    use_dual_scale=_ds, use_virtual_nodes=_vn)

            u_next = u_phys_next + u_corr
            if bnd_info:
                u_next = _apply_bcs(u_next, bnd_info)
            u_hist[:, t + 1] = u_next
            u_curr = u_next.detach()

        out = {"u_final": u_hist}
        if return_anchor_history:
            out["anchor_history"] = anchor_history          # list[T-1] of LongTensor[m]
            out["nodes"] = nodes.detach().cpu()
        return out

    # ── utilities ─────────────────────────────────────────────
    def avg_cg_iters(self):
        """最近一次 forward 中所有（每样本×每时间步）CG 求解的平均迭代次数。
        训练脚本按 batch 累加它做日志；无 CG 调用时返回 0.0。"""
        if self._cg_iter_count == 0:
            return 0.0
        return self._cg_iter_sum / self._cg_iter_count

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def ablation_summary(self):
        anchor_w = self.anchor_selector.weight_summary()
        lines = [
            "=== PhysHGNet Ablation Config ===",
            f"  C1 Physics Anchor : {'ON' if self.use_physics_anchor else 'OFF'}",
            f"  Anchor weights    : {anchor_w}",
            f"  C2 Learned Coarse : {'ON' if self.use_learned_coarse else 'OFF'}",
            f"  C3 Dual-Scale GNN : {'ON' if self.use_dual_scale_gnn else 'OFF'}",
            f"  C3 Virtual Nodes  : {'ON' if self.use_virtual_nodes else 'OFF'}",
            f"  Coarse upd freq   : every {self.coarse_upd_freq} step(s)",
            f"  Total Parameters  : {self.num_parameters():,}",
        ]
        return "\n".join(lines)
