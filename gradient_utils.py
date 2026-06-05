"""
gradient_utils.py — Compute per-node physical gradient norms on 2D meshes.

Provides a single public function:

    compute_gradient_norm(nodes, u, faces, eps=1e-8) -> Tensor [N]

Algorithm (area-weighted finite difference on triangles):
  For each triangle f = (i, j, k) with 2-D coords and scalar field u:
    e1 = p_j - p_i,  e2 = p_k - p_i
    du1 = u_j - u_i, du2 = u_k - u_i
    Solve: [e1|e2]^T G = [du1, du2]^T  (2×2 linear system)
    ||∇u||_f  = ||G||_2           (face gradient norm)
  Scatter area * ||∇u||_f back to every vertex, then divide by total area.

Complexity: O(F), fully vectorised, no Python loops.
Works with nodes.shape[1] == 2 (the laser hardening dataset).
"""

import torch


def compute_gradient_norm(
    nodes: torch.Tensor,   # [N, 2]
    u: torch.Tensor,       # [N] or [N, C]  – physical field (e.g. temperature)
    faces: torch.Tensor,   # [F, 3]         – triangle connectivity
    eps: float = 1e-8,
) -> torch.Tensor:         # [N]            – per-node gradient norm
    """
    Area-weighted mesh gradient norm, compatible with the PhysHGNet pipeline.

    Returns a non-negative tensor of shape [N].  Values are *not* normalised
    (normalisation happens inside PhysicsAwareAnchorSelector.forward()).
    """
    N = nodes.shape[0]
    device = nodes.device

    if u.dim() == 1:
        u = u.unsqueeze(1)          # [N, 1]
    C = u.shape[1]

    vi = faces[:, 0]                # [F]
    vj = faces[:, 1]
    vk = faces[:, 2]

    pi = nodes[vi]                  # [F, 2]
    pj = nodes[vj]
    pk = nodes[vk]

    e1 = pj - pi                   # [F, 2]
    e2 = pk - pi                   # [F, 2]

    # Signed 2-D area
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]   # [F]
    area  = 0.5 * cross.abs()                              # [F]

    # 2×2 system:  [[e1x, e2x],[e1y, e2y]]^T G = [du1, du2]^T
    # det = e1x*e2y - e1y*e2x = cross
    det = cross.clamp(min=eps) if (cross >= 0).all() else \
        torch.where(cross >= 0, cross.clamp(min=eps),
                    cross.clamp(max=-eps))                  # [F]

    du1 = u[vj] - u[vi]           # [F, C]
    du2 = u[vk] - u[vi]           # [F, C]

    # Inverse of [[e1x, e2x],[e1y, e2y]] (transposed edge matrix):
    #   G_x = ( e2y * du1 - e2x * du2) / det
    #   G_y = (-e1y * du1 + e1x * du2) / det
    e1x, e1y = e1[:, 0:1], e1[:, 1:2]    # [F, 1]
    e2x, e2y = e2[:, 0:1], e2[:, 1:2]

    inv_det = 1.0 / det.unsqueeze(1)      # [F, 1]
    Gx = ( e2y * du1 - e2x * du2) * inv_det   # [F, C]
    Gy = (-e1y * du1 + e1x * du2) * inv_det   # [F, C]

    # Face gradient norm: sqrt(Gx^2 + Gy^2) summed over channels
    grad_norm_f = (Gx ** 2 + Gy ** 2).sum(dim=1).sqrt()   # [F]

    # Area-weighted scatter to vertices
    w = area * grad_norm_f                         # [F]
    grad_sum  = torch.zeros(N, device=device, dtype=nodes.dtype)
    area_sum  = torch.zeros(N, device=device, dtype=nodes.dtype)

    for v in [vi, vj, vk]:
        grad_sum.scatter_add_(0, v, w)
        area_sum.scatter_add_(0, v, area)

    return grad_sum / area_sum.clamp(min=eps)      # [N]
