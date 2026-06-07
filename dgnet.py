"""
dgnet.py — DGNet: Discrete Green Network
与仓库 models.py / physics.py 完全对齐

架构 (参见论文 arXiv 2603.01762, §3.4):

  1. L_phys  = build_operator(nodes, edges, faces, vol, 'laplace')
               [N,N] NSD (负半正定): physics.py 返回 −κL_positive
  2. delta_L = OperatorCorrector(graph_data)   [N,N] 微小 Laplacian 修正 (1e-5 量级)
  3. L_total = L_phys + delta_L

  4. CN 递推 (NSD 算子约定 A = I − dt/2 · L_total):
       A · u_{n+1} = B · u_n + dt/2·(f_n+f_{n+1}) + dt·r_nl(u_n)
       A = I − dt/2 · L_total   (PD, 稳定)
       B = I + dt/2 · L_total
       r_nl = NonlinearDynamicsSolver.forward_batched(...)

  5. 残差修正:
       u_{n+1} += ResidualSolver.forward_batched(...)

  6. 密集 LU 分解仅在每次 forward 调用开始时做 **一次**（L_total 不随时间步变化），
     所有 T-1 步共享同一个 (lu, piv)——论文 Appendix A.3 "efficient implementation"。

参数量（默认配置 hidden_dim=128, layers=5 for NL/Res; 64/3 for OpCorr）≈ 568K。
"""

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import build_operator, apply_bcs_to_state
from models import OperatorCorrector, NonlinearDynamicsSolver, ResidualSolver


# ─────────────────────────────────────────────────────────────
# DGNet
# ─────────────────────────────────────────────────────────────
class DGNet(nn.Module):
    """DGNet 主体：三模块混合架构 + Crank–Nicolson 密集 LU。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        sd  = config.get("spatial_dim",         2)
        fd  = config.get("feature_dim",         1)
        od  = config.get("output_dim",          1)
        hd  = config.get("residual_hidden_dim", 128)
        nl  = config.get("residual_num_layers", 5)
        ohd = config.get("operator_hidden_dim", 64)
        onl = config.get("operator_num_layers", 3)
        self.operator_type = config.get("operator_type", "laplace")

        self.operator_corrector  = OperatorCorrector(sd,  ohd, onl)
        self.nonlinear_solver    = NonlinearDynamicsSolver(sd, fd, od, hd, nl)
        self.residual_solver     = ResidualSolver(sd, fd, od, hd, nl)

    # ── 工具 ──────────────────────────────────────────────────
    @staticmethod
    def _bi_edge(edges: torch.Tensor) -> torch.Tensor:
        """单向边 [E,2] → 双向边索引 [2, 2E]"""
        return torch.cat([edges, edges.flip(1)], dim=0).T

    # ── 主前向 ───────────────────────────────────────────────
    def forward(
        self,
        batch: Dict[str, Any],
        use_physics_path: bool = True,
        use_nn_correction: bool = True,
    ) -> Dict[str, torch.Tensor]:

        nodes        = batch["nodes"]                        # [N, d]
        edges        = batch["edges"]                        # [E, 2] unique
        faces        = batch["faces"]                        # [F, 3]
        node_volumes = batch["node_volumes"]                 # [N]
        node_type    = batch.get(
            "node_type",
            torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))
        src_terms    = batch["source_terms"]                 # [B, T, N, C]
        time_pts     = batch["time_points"]                  # [T]
        ic           = batch["initial_conditions"]           # [B, N, C]
        boundary_info = batch.get("boundary_info", {})

        u_curr = ic[:, 0] if ic.dim() == 4 else ic          # [B, N, C]
        B, T, N, C = src_terms.shape
        device = nodes.device
        dt = float((time_pts[1] - time_pts[0]).item()) if T > 1 else 0.01

        bi_e = self._bi_edge(edges)                          # [2, 2E]

        # ── 1. 物理先验算子 L_phys (NSD) ─────────────────────
        L_phys = build_operator(
            nodes, edges, faces, node_volumes,
            operator_type=self.operator_type)                # [N,N] NSD

        # ── 2. 算子修正 delta_L ───────────────────────────────
        if use_nn_correction:
            graph_data = {
                "nodes":        nodes,
                "edges":        edges,
                "node_volumes": node_volumes,
                "node_type":    node_type,
            }
            delta_L = self.operator_corrector(graph_data)   # [N,N] ~1e-5
            L_total = L_phys + delta_L
        else:
            L_total = L_phys

        # ── 3. CN 矩阵  (NSD 约定: A = I − dt/2·L) ───────────
        # L_total ≈ L_phys = −κ L_positive  (NSD)
        # A = I − dt/2·L  ≡ I + dt/2·κ·L_pos  (正定, 稳定)
        # B = I + dt/2·L  ≡ I − dt/2·κ·L_pos  (特征值 ≤ 1, 有界)
        #
        # 全程 float64 避免稠密矩阵乘法中的 catastrophic cancellation
        Ld  = L_total.double()
        I64 = torch.eye(N, device=device, dtype=torch.float64)
        A   = (I64 - (dt / 2.0) * Ld).contiguous()         # [N,N] f64
        B_op = I64 + (dt / 2.0) * Ld                        # [N,N] f64
        lu, piv = torch.linalg.lu_factor(A)                  # 密集 LU，仅一次

        # ── 4. 时间循环 ───────────────────────────────────────
        u_hist = torch.zeros(B, T, N, C, device=device)
        u_hist[:, 0] = u_curr

        for t in range(T - 1):
            f_cur  = src_terms[:, t]                         # [B, N, C]
            f_next = src_terms[:, t + 1]

            # 4a. 非线性动力学修正 r_nl (显式, 在 u_n 处估值)
            if use_nn_correction:
                r_nl = self.nonlinear_solver.forward_batched(
                    nodes, bi_e, u_curr, node_type)          # [B, N, C]
            else:
                r_nl = torch.zeros_like(u_curr)

            # 4b. RHS = B·u_n + dt/2·(f_n+f_{n+1}) + dt·r_nl
            u2d   = u_curr.permute(1, 0, 2).reshape(N, B * C).double()
            Bu    = (B_op @ u2d).reshape(N, B, C).permute(1, 0, 2)  # f64
            rhs   = (Bu
                     + (dt / 2.0) * (f_cur + f_next).double()
                     + dt * r_nl.double())                   # [B,N,C] f64
            rhs2d = rhs.permute(1, 0, 2).reshape(N, B * C)

            # 4c. A·u_{n+1} = rhs  →  u_phys
            sol    = torch.linalg.lu_solve(lu, piv, rhs2d).float()
            u_phys = sol.reshape(N, B, C).permute(1, 0, 2)  # [B,N,C]

            # 4d. 残差修正
            if use_nn_correction:
                r_res = self.residual_solver.forward_batched(
                    nodes, bi_e, u_phys, node_type, boundary_info)   # [B,N,C]
            else:
                r_res = torch.zeros_like(u_phys)

            u_next = u_phys + r_res

            # 4e. 边界条件（仅 Dirichlet 非空时生效）
            if boundary_info:
                di = boundary_info.get("dirichlet")
                if di is not None and len(di.get("indices", [])) > 0:
                    idx = di["indices"].to(device)
                    val = di["values"].to(device)
                    u_next[:, idx, :] = val.unsqueeze(0).unsqueeze(-1).expand(
                        B, -1, C)

            u_hist[:, t + 1] = u_next
            u_curr = u_next.detach()

        return {"u_final": u_hist}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────────────────────
class DGNetLoss(nn.Module):
    """首步 + 末步 MSE，与 PhysHGNet 训练目标一致。"""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

    def forward(self, predictions: Dict, targets: torch.Tensor) -> Dict:
        u  = predictions["u_final"]
        l1 = F.mse_loss(u[:, 1],  targets[:, 1])
        lT = F.mse_loss(u[:, -1], targets[:, -1])
        return {"total_loss": l1 + lT,
                "first_step_loss": l1, "final_step_loss": lT}
