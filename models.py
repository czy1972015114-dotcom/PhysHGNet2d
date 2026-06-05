"""
models.py — DGNet 神经模块

与仓库 czy1972015114-dotcom/DGNEt 的 heat_equation_benchmark.py 对齐。

主要变化（相对原版）：
  [1] _OrigMPNN 对齐仓库：message + update 为独立 nn.Sequential，
      不再依赖 torch_geometric.nn.MLP（消除版本兼容问题）。
  [2] OperatorCorrector：去掉 LayerNorm、简化节点编码，与仓库 node_enc 对齐。
  [3] NonlinearDynamicsSolver / ResidualSolver：
      - 内部 MPNN 使用新的 _OrigMPNN，行为与仓库一致。
      - 批量化接口 forward_batched(nodes, bi_e, u_batch, node_type) 为
        DGNet.forward 的 _batch_graphs 路径提供统一入口。
      - 解码器保持零初始化（与仓库 Bug Fix 逻辑一致）。

向后兼容：
  DGNet.forward 仍然可以调用 .forward(graph_data) 形式，
  也可以调用新的 .forward_batched() 形式（DGNet 内部使用后者）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────
# 基础 MPNN（对齐仓库 _OrigMPNN）
# ─────────────────────────────────────────────────────────────

class _OrigMPNN(nn.Module):
    """
    单层 MPNN，与仓库 heat_equation_benchmark.py 的 _OrigMPNN 完全一致。

    message = MLP([x_i, x_j, edge_attr]) → hidden
    update  = MLP([x, aggr_msg])        → node_dim（残差+LayerNorm 由外部处理）
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 aggr: str = 'mean'):
        super().__init__()
        self.aggr = aggr
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim), nn.ReLU())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index   # edge_index: [2, E]
        msg = self.msg_mlp(torch.cat([x[dst], x[src], edge_attr], dim=-1))  # [E, H]

        # 聚合到目标节点
        N = x.shape[0]
        if self.aggr == 'mean':
            agg = torch.zeros(N, msg.shape[-1], device=x.device, dtype=x.dtype)
            cnt = torch.zeros(N, 1,             device=x.device, dtype=x.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            cnt.scatter_add_(0, dst.unsqueeze(1),
                             torch.ones(len(dst), 1, device=x.device, dtype=x.dtype))
            agg = agg / cnt.clamp(min=1)
        else:  # sum
            agg = torch.zeros(N, msg.shape[-1], device=x.device, dtype=x.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

        return self.upd_mlp(torch.cat([x, agg], dim=-1))   # [N, node_dim]


# ─────────────────────────────────────────────────────────────
# OperatorCorrector（对齐仓库 node_enc / edge_corr 结构）
# ─────────────────────────────────────────────────────────────

class OperatorCorrector(nn.Module):
    """
    学习图边上的算子修正 ΔL（稠密 N×N，_corr_scale=1e-5）。

    与仓库对齐：
      - node_enc: Linear(spatial+1+ntype, hd) + ReLU + Linear
      - op_edge_enc: Linear(spatial+1, hd) + ReLU + Linear
      - op_mp: num_layers 个 _OrigMPNN + LayerNorm（残差）
      - edge_corr: Linear(2*hd, hd) + ReLU + Linear(hd, 1)
      - 最终输出 = _corr_scale * tanh(raw)
    """

    def __init__(self, spatial_dim: int, hidden_dim: int = 64,
                 num_layers: int = 3, num_node_types: int = 3):
        super().__init__()
        self.spatial_dim    = spatial_dim
        self.num_node_types = num_node_types
        self._corr_scale    = 1e-5

        # 节点编码：与仓库 node_enc 一致
        self.node_enc = nn.Sequential(
            nn.Linear(spatial_dim + 1 + num_node_types, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        # 边编码：[Δx, ‖Δx‖] → hidden
        self.op_edge_enc = nn.Sequential(
            nn.Linear(spatial_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        # MPNN 层（与仓库 op_mp / op_norms 对齐）
        self.op_mp    = nn.ModuleList(
            [_OrigMPNN(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.op_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # 边修正预测
        self.edge_corr = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, graph_data: Dict[str, Any]) -> torch.Tensor:
        nodes        = graph_data['nodes']                          # [N, d]
        unique_edges = graph_data['edges']                          # [E, 2]
        node_volumes = graph_data.get(
            'node_volumes', torch.ones(nodes.shape[0], device=nodes.device))
        node_type    = graph_data.get(
            'node_type',    torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))

        N      = nodes.shape[0]
        device = nodes.device

        # 双向边
        bi_edges  = torch.cat([unique_edges, unique_edges.flip(1)], dim=0)  # [2E, 2]
        edge_index = bi_edges.T                                              # [2, 2E]

        # 边特征
        diff = nodes[bi_edges[:, 1]] - nodes[bi_edges[:, 0]]
        ea   = self.op_edge_enc(torch.cat([diff, diff.norm(dim=-1, keepdim=True)], dim=-1))

        # 节点特征
        nt = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        h  = self.node_enc(torch.cat([nodes, node_volumes.unsqueeze(-1), nt], dim=-1))

        # MPNN（残差 + LayerNorm，与仓库 _run_mpnn 一致）
        for layer, norm in zip(self.op_mp, self.op_norms):
            h = norm(h + layer(h, edge_index, ea))

        # 边修正
        src, dst = edge_index
        raw  = self.edge_corr(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)  # [2E]
        corr = self._corr_scale * torch.tanh(raw)

        # 组装稠密 ΔL（行和=0，即 ΔL 是 Laplacian 形式）
        delta_L = torch.zeros(N, N, device=device)
        delta_L[dst, src] = corr
        delta_L.fill_diagonal_(0)
        delta_L -= torch.diag(delta_L.sum(dim=1))
        return delta_L


# ─────────────────────────────────────────────────────────────
# 批量化辅助：将 B 个同拓扑图拼为一个大图
# ─────────────────────────────────────────────────────────────

def batch_graphs(nodes: torch.Tensor, bi_e: torch.Tensor,
                 u_batch: torch.Tensor, node_type: torch.Tensor):
    """
    与仓库 OriginalDGNet._batch_graphs 完全对齐。

    Args:
        nodes:     [N, d]
        bi_e:      [2, E]  双向边索引（统一格式 [src, dst]）
        u_batch:   [B, N, C]
        node_type: [N]

    Returns:
        nodes_b:   [B*N, d]
        bi_e_b:    [2, B*E]  加偏移后的边索引
        u_flat:    [B*N, C]
        nt_b:      [B*N, 3]  one-hot node type
    """
    B, N, C = u_batch.shape
    E       = bi_e.shape[1]
    device  = nodes.device
    offsets = torch.arange(B, device=device) * N     # [B]

    # 边索引：第 b 个图偏移 b*N
    bi_e_b = (bi_e.unsqueeze(0) +                    # [1, 2, E]
              offsets.view(B, 1, 1)                   # [B, 1, 1]
              ).permute(1, 0, 2).reshape(2, B * E)    # [2, B*E]

    nodes_b = nodes.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
    nt_b    = F.one_hot(node_type.long(), num_classes=3).float() \
                .unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
    u_flat  = u_batch.reshape(B * N, C)

    return nodes_b, bi_e_b, u_flat, nt_b


# ─────────────────────────────────────────────────────────────
# NonlinearDynamicsSolver（对齐仓库 _compute_nonlinear_batched）
# ─────────────────────────────────────────────────────────────

class NonlinearDynamicsSolver(nn.Module):
    """
    预测非线性动力学项 r(u^k)。

    forward(graph_data)         → 单样本接口（向后兼容）
    forward_batched(nodes, bi_e, u_batch, node_type) → 批量接口（DGNet 内部使用）
    """

    def __init__(self, spatial_dim: int, node_feature_dim: int, output_dim: int,
                 hidden_dim: int = 128, num_processing_layers: int = 5,
                 num_node_types: int = 3):
        super().__init__()
        self.spatial_dim    = spatial_dim
        self.output_dim     = output_dim
        self.num_node_types = num_node_types
        edge_dim = hidden_dim // 8

        self.nl_node_enc = nn.Sequential(
            nn.Linear(node_feature_dim + spatial_dim + num_node_types, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.nl_edge_enc = nn.Sequential(
            nn.Linear(spatial_dim + node_feature_dim + 1, edge_dim), nn.ReLU())
        self.nl_mp    = nn.ModuleList(
            [_OrigMPNN(hidden_dim, edge_dim, hidden_dim) for _ in range(num_processing_layers)])
        self.nl_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_processing_layers)])
        self.nl_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        # 零初始化（与仓库一致）
        nn.init.zeros_(self.nl_dec[-1].weight)
        nn.init.zeros_(self.nl_dec[-1].bias)

    def _run(self, h: torch.Tensor, edge_index: torch.Tensor,
             ea: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.nl_mp, self.nl_norms):
            h = norm(h + layer(h, edge_index, ea))
        return self.nl_dec(h)

    def forward_batched(self, nodes: torch.Tensor, bi_e: torch.Tensor,
                        u_batch: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """批量接口，与仓库 _compute_nonlinear_batched 完全对齐。"""
        B, N, C = u_batch.shape
        nodes_b, bi_e_b, u_flat, nt_b = batch_graphs(nodes, bi_e, u_batch, node_type)

        diff = nodes_b[bi_e_b[1]] - nodes_b[bi_e_b[0]]
        fd   = u_flat[bi_e_b[1]] - u_flat[bi_e_b[0]]
        ea   = self.nl_edge_enc(torch.cat([diff, fd, diff.norm(dim=-1, keepdim=True)], dim=-1))
        h    = self.nl_node_enc(torch.cat([u_flat, nodes_b, nt_b], dim=-1))
        return self._run(h, bi_e_b, ea).reshape(B, N, C)

    def forward(self, graph_data: Dict[str, Any]) -> torch.Tensor:
        """单样本接口，向后兼容。"""
        nodes     = graph_data['nodes']
        u         = graph_data['node_features']
        node_type = graph_data.get(
            'node_type', torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))
        edges = graph_data['edges']
        bi_e  = torch.cat([edges, edges.flip(1)], dim=0).T
        return self.forward_batched(nodes, bi_e, u.unsqueeze(0), node_type).squeeze(0)


# ─────────────────────────────────────────────────────────────
# ResidualSolver（对齐仓库 _compute_residual_batched）
# ─────────────────────────────────────────────────────────────

class ResidualSolver(nn.Module):
    """
    残差修正求解器，Dirichlet 节点在每层后恢复锚点特征。

    forward(graph_data)         → 单样本接口（向后兼容）
    forward_batched(nodes, bi_e, u_batch, node_type, boundary_info)
                                → 批量接口（DGNet 内部使用）
    """

    def __init__(self, spatial_dim: int, node_feature_dim: int, output_dim: int,
                 hidden_dim: int = 128, num_processing_layers: int = 5,
                 num_node_types: int = 3):
        super().__init__()
        self.spatial_dim          = spatial_dim
        self.output_dim           = output_dim
        self.num_node_types       = num_node_types
        self.num_processing_layers = num_processing_layers
        edge_dim = hidden_dim // 8

        self.res_node_enc = nn.Sequential(
            nn.Linear(node_feature_dim + spatial_dim + num_node_types, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.res_edge_enc = nn.Sequential(
            nn.Linear(spatial_dim + node_feature_dim + 1, edge_dim), nn.ReLU())
        self.res_mp    = nn.ModuleList(
            [_OrigMPNN(hidden_dim, edge_dim, hidden_dim) for _ in range(num_processing_layers)])
        self.res_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_processing_layers)])
        self.res_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        # 零初始化（与仓库一致）
        nn.init.zeros_(self.res_dec[-1].weight)
        nn.init.zeros_(self.res_dec[-1].bias)

    def forward_batched(self, nodes: torch.Tensor, bi_e: torch.Tensor,
                        u_batch: torch.Tensor, node_type: torch.Tensor,
                        boundary_info: Optional[Dict] = None) -> torch.Tensor:
        """批量接口，与仓库 _compute_residual_batched 完全对齐。"""
        B, N, C = u_batch.shape
        device  = nodes.device
        nodes_b, bi_e_b, u_flat, nt_b = batch_graphs(nodes, bi_e, u_batch, node_type)

        diff = nodes_b[bi_e_b[1]] - nodes_b[bi_e_b[0]]
        fd   = u_flat[bi_e_b[1]] - u_flat[bi_e_b[0]]
        ea   = self.res_edge_enc(torch.cat([diff, fd, diff.norm(dim=-1, keepdim=True)], dim=-1))
        h    = self.res_node_enc(torch.cat([u_flat, nodes_b, nt_b], dim=-1))
        h_anc = h.clone()

        # Dirichlet 节点在大图中的绝对索引（每个子图偏移 b*N）
        dir_idx = None
        if boundary_info and isinstance(boundary_info, dict):
            di = boundary_info.get('dirichlet')
            if di is not None:
                idx = di.get('indices')
                if idx is not None and len(idx) > 0:
                    offsets  = torch.arange(B, device=device) * N
                    dir_idx  = torch.cat([idx.to(device) + off for off in offsets], dim=0)

        for layer, norm in zip(self.res_mp, self.res_norms):
            h = norm(h + layer(h, bi_e_b, ea))
            if dir_idx is not None:
                h = h.clone()
                h[dir_idx] = h_anc[dir_idx]

        return self.res_dec(h).reshape(B, N, C)

    def forward(self, graph_data: Dict[str, Any]) -> torch.Tensor:
        """单样本接口，向后兼容。"""
        nodes         = graph_data['nodes']
        u             = graph_data['node_features']
        boundary_info = graph_data.get('boundary_info', {})
        node_type     = graph_data.get(
            'node_type', torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device))
        edges = graph_data['edges']
        bi_e  = torch.cat([edges, edges.flip(1)], dim=0).T
        return self.forward_batched(nodes, bi_e, u.unsqueeze(0),
                                    node_type, boundary_info).squeeze(0)
