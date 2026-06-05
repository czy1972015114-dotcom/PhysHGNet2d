"""
structured_models.py — GNN-parameterized multiscale structured operator.

L_neural = alpha_loc * S_theta^loc + alpha_coarse * P_theta C_theta R_theta

Memory: O(E + N*m) instead of O(N^2).
All graph utilities are O(E) or O(N log N) — never O(N^2).

Source: PhysHGNet repo (unchanged).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, aggr='mean'):
        super().__init__(aggr=aggr)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class MPNNProcessor(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])

    def forward(self, h, edge_index, edge_attr):
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(h, edge_index, edge_attr)
            h = norm(h)
        return h


class FineGraphEncoder(nn.Module):
    def __init__(self, spatial_dim, hidden_dim, num_layers, num_node_types=3):
        super().__init__()
        node_in = spatial_dim + 1 + num_node_types
        edge_in = spatial_dim + 1
        self.node_enc = nn.Sequential(
            nn.Linear(node_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.processor = MPNNProcessor(hidden_dim, hidden_dim, hidden_dim, num_layers)

    def forward(self, nodes, edge_index, edge_attr, node_volumes, node_type,
                num_node_types=3):
        nt = F.one_hot(node_type.long(), num_classes=num_node_types).float()
        x = self.node_enc(torch.cat([nodes, node_volumes.unsqueeze(-1), nt], -1))
        ea = self.edge_enc(edge_attr)
        return self.processor(x, edge_index, ea)


class LocalCorrectionHead(nn.Module):
    def __init__(self, node_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, h, edge_index, edge_attr, reverse_map):
        src, dst = edge_index
        raw = self.psi(torch.cat([h[src], h[dst], edge_attr], -1)).squeeze(-1)
        return F.softplus((raw + raw[reverse_map]) / 2.0)


def farthest_point_sampling(coords, m):
    N = coords.shape[0]
    device = coords.device
    dists = torch.full((N,), float('inf'), device=device)
    sel = []
    idx = 0
    for _ in range(m):
        sel.append(idx)
        d = (coords - coords[idx]).norm(dim=-1)
        dists = torch.minimum(dists, d)
        dists[idx] = -1.0
        idx = dists.argmax().item()
    return torch.tensor(sel, dtype=torch.long, device=device)


class ProlongationNet(nn.Module):
    def __init__(self, node_dim, hidden_dim, spatial_dim=2, q=4, tau=0.1):
        super().__init__()
        self.q, self.tau = q, tau
        self.score_net = nn.Sequential(
            nn.Linear(2 * node_dim + spatial_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, h_fine, h_coarse, coords, anchor_coords):
        N, m, d = coords.shape[0], anchor_coords.shape[0], coords.shape[1]
        device = coords.device
        q = min(self.q, m)

        diff = coords.unsqueeze(1) - anchor_coords.unsqueeze(0)  # [N, m, d]
        dist_sq = (diff ** 2).sum(-1)
        _, topq = dist_sq.topk(q, dim=1, largest=False)

        geo = -dist_sq.gather(1, topq) / self.tau
        h_i = h_fine.unsqueeze(1).expand(-1, q, -1)
        h_k = h_coarse[topq]
        d_q = diff.gather(1, topq.unsqueeze(-1).expand(-1, -1, d))
        neural = self.score_net(torch.cat([h_i, h_k, d_q], -1)).squeeze(-1)

        w = torch.softmax(geo + neural, dim=1)
        P = torch.zeros(N, m, device=device)
        P.scatter_(1, topq, w)
        col_sum = P.sum(0).clamp(min=1e-6)
        R = (P / col_sum.unsqueeze(0)).t()
        return P, R


class CoarseGraphModule(nn.Module):
    def __init__(self, node_dim, edge_feat_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.edge_enc = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU())
        self.processor = MPNNProcessor(node_dim, hidden_dim, hidden_dim, num_layers)
        self.head = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, g, coarse_ei, coarse_ea, coarse_rev):
        ee = self.edge_enc(coarse_ea)
        g = self.processor(g, coarse_ei, ee)
        src, dst = coarse_ei
        raw = self.head(torch.cat([g[src], g[dst], coarse_ea], -1)).squeeze(-1)
        return F.softplus((raw + raw[coarse_rev]) / 2.0), g


def sparse_laplacian_matvec(weights, edge_index, x, N):
    """(L x)_i = sum_j w_ij (x_j - x_i). O(E)."""
    weights = weights.float()
    x = x.float()
    src, dst = edge_index
    if x.dim() == 1:
        msg = weights * (x[dst] - x[src])
        out = torch.zeros(N, device=x.device, dtype=torch.float32)
        out.scatter_add_(0, src, msg)
    elif x.dim() == 2:
        msg = weights.unsqueeze(-1) * (x[dst] - x[src])
        out = torch.zeros_like(x)
        out.scatter_add_(0, src.unsqueeze(-1).expand_as(msg), msg)
    else:
        raise ValueError(f"x.dim()={x.dim()} not supported")
    return out


def structured_L_matvec(x, Lp_w, fine_ei, s_w, a_loc,
                         P, R, c_w, coarse_ei, a_coarse, N, m):
    """L_theta x = L_phys x + alpha_loc * S x + alpha_coarse * P C R x."""
    x = x.float()
    out = sparse_laplacian_matvec(Lp_w, fine_ei, x, N)
    out = out + a_loc * sparse_laplacian_matvec(s_w, fine_ei, x, N)

    P_f, R_f = P.float(), R.float()
    Rx = R_f @ (x.unsqueeze(-1) if x.dim() == 1 else x)
    if Rx.dim() == 2 and Rx.shape[-1] == 1:
        Rx = Rx.squeeze(-1)
    CRx = sparse_laplacian_matvec(c_w, coarse_ei, Rx, m)
    PCRx = P_f @ (CRx.unsqueeze(-1) if CRx.dim() == 1 else CRx)
    if PCRx.dim() == 2 and PCRx.shape[-1] == 1:
        PCRx = PCRx.squeeze(-1)
    out = out + a_coarse * PCRx
    return out


def build_bidirectional_edges(unique_edges):
    """[E_uniq, 2] -> [2, 2E] bidirectional."""
    return torch.cat([unique_edges, unique_edges.flip(1)], 0).T


def build_edge_features(nodes, edge_index):
    """[E, d+1]: (coord_diff, distance)."""
    diff = nodes[edge_index[1]] - nodes[edge_index[0]]
    return torch.cat([diff, diff.norm(dim=-1, keepdim=True)], -1)


def build_reverse_edge_map(edge_index):
    E = edge_index.shape[1]
    src, dst = edge_index[0].long(), edge_index[1].long()
    N = max(src.max(), dst.max()) + 1
    fwd = src * N + dst
    rev = dst * N + src
    sorted_fwd, perm = fwd.sort()
    pos = torch.searchsorted(sorted_fwd, rev).clamp(max=E - 1)
    return perm[pos]


def build_knn_graph(coords, k):
    """Symmetric kNN graph for coarse (anchor) nodes."""
    N, device = coords.shape[0], coords.device
    k = min(k, N - 1)
    dist = torch.cdist(coords, coords)
    dist.fill_diagonal_(float('inf'))
    _, knn = dist.topk(k, dim=1, largest=False)
    src = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = knn.reshape(-1)
    both = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    key = both[0] * N + both[1]
    ukey = key.unique(sorted=True)
    ei = torch.stack([ukey // N, ukey % N]).long().to(device)
    ea = build_edge_features(coords, ei)
    return ei, ea
