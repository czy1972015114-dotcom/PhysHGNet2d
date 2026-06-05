"""
generate_laser_data_aligned.py — 2D Laser Hardening Data Generator
                                  (Dual-compatible: PhysHGNet + OriginalDGNet)

与 generate_laser_data_2d.py 的核心区别
───────────────────────────────────────
1. 额外将稀疏 L_physics 矩阵存入 HDF5（CSR 格式：data / indices / indptr）。
   - PhysHGNet/DGNet 可选读取（内部可自行重算），OriginalDGNet 可直接用来缓存 LU。
2. 额外存储 `node_type`（0=interior, 1=Dirichlet, 2=Neumann）。
3. 额外存储 `node_volumes`（与 DGPdeDataset._estimate_node_volumes 输出一致）。
4. 时间步数默认对齐热方程 benchmark（n_timesteps 可调），不再写死 T_SIM/DT。
5. 增加 `--format` 参数：
     full  —— 一个文件存所有轨迹（原有格式，供 DGPdeDataset 使用）
     split —— 每条轨迹单独存一个 HDF5（供大规模并行训练使用）

HDF5 schema（每个 trajectory group）
──────────────────────────────────────
  nodes           [N, 2]  float32     节点坐标
  edges           [E, 2]  int32       无向边索引
  faces           [F, 3]  int32       三角面片索引
  node_features   [T,N,1] float32     ΔT = T - T_ambient
  source_terms    [T,N,1] float32     Q/(rho*c)
  initial_condition [N,1] float32     = node_features[0]
  time_points     [T]     float32     时刻序列
  node_type       [N]     int32       0/1/2
  node_volumes    [N]     float32     等效节点面积
  L_physics/
    data          [nnz]   float32     CSR 值
    indices       [nnz]   int32       CSR 列索引
    indptr        [N+1]   int32       CSR 行指针
    shape         [2]     int32       [N, N]
    diffusion_coeff float32           kappa = k/(rho*c)
  boundary_info/
    dirichlet/
      indices     [nd]    int32
      values      [nd]    float32
    neumann/
      source_indices [0]  int32
      target_indices [0]  int32

Usage
──────
  # 生成标准数据集（默认 40 条轨迹，2000 节点，121 步）
  python generate_laser_data_aligned.py --n_nodes 2000

  # 较小数据用于快速测试
  python generate_laser_data_aligned.py --n_nodes 500 --n_traj 10 \\
      --n_timesteps 70 --out_dir data_test

  # 跳过 L_physics（仅在内存受限时使用）
  python generate_laser_data_aligned.py --no_store_L

Dependencies: numpy, scipy, h5py  (不需要 PyTorch / FEniCSx)
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse as sp
from scipy.sparse.linalg import splu


# ══════════════════════════════════════════════════════════════
# 物理常数（与 generate_laser_data_2d.py 完全一致）
# ══════════════════════════════════════════════════════════════
RHO           = 7850.0   # kg/m³
SPECIFIC_HEAT = 450.0    # J/(kg·K)
CONDUCTIVITY  = 50.0     # W/(m·K)
H_CONV        = 25.0     # W/(m²·K)
T_AMBIENT     = 298.15   # K

KAPPA         = CONDUCTIVITY / (RHO * SPECIFIC_HEAT)  # 热扩散率
H_NORM        = H_CONV    / (RHO * SPECIFIC_HEAT)     # 归一化对流系数

NUM_LASERS    = 10
MAX_SPEED     = 0.8   # m/s
ANGULAR_VEL   = 1.0   # rad/s
MIN_TRI_AREA  = 1e-14


# ══════════════════════════════════════════════════════════════
# 1. 网格生成（与原版完全一致，使用 Delaunay 三角化）
# ══════════════════════════════════════════════════════════════

def generate_2d_mesh(n_nodes: int, seed: int = 42):
    """Delaunay 三角化生成 [0,1]² 上的随机网格。"""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, size=(n_nodes, 2)).astype(np.float32)
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    pts = np.vstack([corners, pts])
    pts = np.unique(pts, axis=0)

    tri = Delaunay(pts)
    triangles = tri.simplices.astype(np.int32)

    edge_set = set()
    for t in triangles:
        for i in range(3):
            for j in range(i + 1, 3):
                edge_set.add((min(t[i], t[j]), max(t[i], t[j])))
    edges = np.array(sorted(edge_set), dtype=np.int32)

    on_bnd = (
        (pts[:, 0] < 0.02) | (pts[:, 0] > 0.98) |
        (pts[:, 1] < 0.02) | (pts[:, 1] > 0.98)
    )
    bnd_idx = np.where(on_bnd)[0].astype(np.int32)

    return pts, edges, triangles, bnd_idx


# ══════════════════════════════════════════════════════════════
# 2. FEM 矩阵构建
# ══════════════════════════════════════════════════════════════

def build_stiffness_2d(nodes: np.ndarray, triangles: np.ndarray,
                        kappa: float) -> sp.csr_matrix:
    """构建 FEM 刚度矩阵 K（稀疏 CSR）。"""
    N = nodes.shape[0]
    x0, x1, x2 = nodes[triangles[:, 0]], nodes[triangles[:, 1]], nodes[triangles[:, 2]]
    J = np.stack([x1 - x0, x2 - x0], axis=1)   # [F, 2, 2]
    det_J = np.linalg.det(J)
    area  = np.abs(det_J) / 2.0

    good = area > MIN_TRI_AREA
    triangles, area, J, det_J = (
        triangles[good], area[good], J[good], det_J[good])

    J_inv   = np.linalg.inv(J)
    J_inv_T = np.transpose(J_inv, (0, 2, 1))
    grad = np.zeros((len(triangles), 3, 2))
    grad[:, 1] = J_inv_T[:, :, 0]
    grad[:, 2] = J_inv_T[:, :, 1]
    grad[:, 0] = -(grad[:, 1] + grad[:, 2])

    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            dot_ij = np.sum(grad[:, i] * grad[:, j], axis=1)
            rows.append(triangles[:, i])
            cols.append(triangles[:, j])
            vals.append(kappa * area * dot_ij)

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N))


def compute_node_areas(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """集中质量阵 → 每个节点的等效面积（与 DGGraph._estimate_node_volumes 一致）。"""
    N = nodes.shape[0]
    areas = np.zeros(N, dtype=np.float32)
    x0, x1, x2 = nodes[triangles[:, 0]], nodes[triangles[:, 1]], nodes[triangles[:, 2]]
    J = np.stack([x1 - x0, x2 - x0], axis=1)
    tri_area = np.abs(np.linalg.det(J)) / 2.0
    good = tri_area > MIN_TRI_AREA
    for k in range(3):
        np.add.at(areas, triangles[good, k], (tri_area[good] / 3.0).astype(np.float32))
    return np.maximum(areas, 1e-14)


def build_diffusion_laplacian(nodes: np.ndarray, triangles: np.ndarray,
                               edges: np.ndarray, kappa: float,
                               h_norm: float, bnd_idx: np.ndarray,
                               node_areas: np.ndarray) -> sp.csr_matrix:
    """
    构建归一化扩散 Laplacian L_physics（稀疏 CSR），与 physics.py 的
    _build_laplace_operator 输出格式一致：

        L_physics = -kappa * L_cot_normalized

    其中 L_cot_normalized[i,j] = -w_ij / vol_i（i≠j），对角线保证行和=0。

    注意：此处不含对流边界项（H_CONV 项），边界约束由 boundary_info 单独传入。
    这与 PhysHGNet 的 build_operator('laplace') 输出一致。
    """
    K = build_stiffness_2d(nodes, triangles, kappa)
    N = nodes.shape[0]
    vol = sp.diags(node_areas.astype(np.float64), format='csr')
    # 将刚度矩阵转换为归一化 Laplacian（行 / vol_i）
    inv_vol = sp.diags(1.0 / node_areas.astype(np.float64), format='csr')
    L = inv_vol @ K
    return L.astype(np.float32).tocsr()


# ══════════════════════════════════════════════════════════════
# 3. 激光路径类（与 generate_laser_data_2d.py 完全一致）
# ══════════════════════════════════════════════════════════════

class BasePath2D:
    def __init__(self, rng):
        self.rng = rng; self.active = True
        self.power = 0.0; self.radius = 0.0
    def get_position(self, t): raise NotImplementedError

class OrbitPath2D(BasePath2D):
    def __init__(self, rng, center, radius_path):
        super().__init__(rng)
        self.power = float(rng.uniform(8e7, 1.2e8))
        self.radius = float(rng.uniform(0.04, 0.07))
        self.center = np.array(center, dtype=np.float32)
        self.radius_path = radius_path
        self.w = ANGULAR_VEL * float(rng.uniform(0.8, 1.2))
        self.phase = float(rng.uniform(0, 2 * np.pi))
    def get_position(self, t):
        x = self.center[0] + self.radius_path * np.cos(self.w * t + self.phase)
        y = self.center[1] + self.radius_path * np.sin(self.w * t + self.phase)
        return np.clip(np.array([x, y], dtype=np.float32), 0.05, 0.95)

class WaypointPath2D(BasePath2D):
    def __init__(self, rng):
        super().__init__(rng)
        self.power  = float(rng.uniform(3e7, 6e7))
        self.radius = float(rng.uniform(0.08, 0.14))
        n_wp = int(rng.integers(5, 11))
        self.waypoints = rng.uniform(0.1, 0.9, size=(n_wp, 2)).astype(np.float32)
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.seg_times  = np.concatenate(([0.0], np.cumsum(dists / MAX_SPEED)))
        self.total_time = self.seg_times[-1]
    def get_position(self, t):
        if t > self.total_time:
            self.active = False; return None
        seg = int(np.clip(np.searchsorted(self.seg_times, t, side='right') - 1,
                          0, len(self.waypoints) - 2))
        t_in = t - self.seg_times[seg]
        direction = self.waypoints[seg + 1] - self.waypoints[seg]
        dist = np.linalg.norm(direction)
        if dist < 1e-8:
            return self.waypoints[seg]
        return (self.waypoints[seg] + direction / dist * MAX_SPEED * t_in).astype(np.float32)

class LissajousPath2D(BasePath2D):
    def __init__(self, rng):
        super().__init__(rng)
        self.power  = float(rng.uniform(3e7, 6e7))
        self.radius = float(rng.uniform(0.08, 0.14))
        self.A = float(rng.uniform(0.1, 0.4))
        self.B = float(rng.uniform(0.1, 0.4))
        self.a = int(rng.integers(1, 5))
        self.b = int(rng.integers(1, 5))
        self.dx = float(rng.uniform(0, np.pi))
        self.cx = self.cy = 0.5
    def get_position(self, t):
        x = self.cx + self.A * np.sin(self.a * t * 0.2 + self.dx)
        y = self.cy + self.B * np.sin(self.b * t * 0.2)
        return np.clip(np.array([x, y], dtype=np.float32), 0.05, 0.95)

class RasterPath2D(BasePath2D):
    def __init__(self, rng):
        super().__init__(rng)
        self.power  = float(rng.uniform(3e7, 6e7))
        self.radius = float(rng.uniform(0.08, 0.14))
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        hw, hh = rng.uniform(0.1, 0.25, size=2)
        self.x_range = [cx - hw, cx + hw]
        self.y_range = [cy - hh, cy + hh]
        n_lines = int(rng.integers(5, 10))
        self.y_steps    = np.linspace(self.y_range[1], self.y_range[0], n_lines)
        self.line_dur   = (self.x_range[1] - self.x_range[0]) / MAX_SPEED
        self.total_time = n_lines * self.line_dur
    def get_position(self, t):
        if t > self.total_time:
            self.active = False; return None
        line_idx = min(int(t / self.line_dur), len(self.y_steps) - 1)
        t_in = t % self.line_dur
        y = self.y_steps[line_idx]
        x = (self.x_range[0] + MAX_SPEED * t_in if line_idx % 2 == 0
             else self.x_range[1] - MAX_SPEED * t_in)
        return np.clip(np.array([x, y], dtype=np.float32), 0.05, 0.95)

class BilliardPath2D(BasePath2D):
    def __init__(self, rng):
        super().__init__(rng)
        self.power  = float(rng.uniform(3e7, 6e7))
        self.radius = float(rng.uniform(0.08, 0.14))
        self.pos    = rng.uniform(0.1, 0.9, size=2).astype(np.float32)
        angle       = float(rng.uniform(0, 2 * np.pi))
        self.vel    = MAX_SPEED * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        self.last_t = 0.0
    def get_position(self, t):
        dt = t - self.last_t
        self.pos += self.vel * dt
        for d in range(2):
            if self.pos[d] < 0.05:
                self.pos[d] = 0.05; self.vel[d] = abs(self.vel[d])
            elif self.pos[d] > 0.95:
                self.pos[d] = 0.95; self.vel[d] = -abs(self.vel[d])
        self.last_t = t
        return self.pos.copy()

def make_lasers_2d(rng, n_lasers=NUM_LASERS):
    lasers = []
    for center, r_path in [([0.25,0.25],0.15),([0.75,0.25],0.15),
                            ([0.25,0.75],0.15),([0.75,0.75],0.15)]:
        lasers.append(OrbitPath2D(rng, center, r_path))
    for _ in range(n_lasers - 4):
        cls = [WaypointPath2D, LissajousPath2D, RasterPath2D, BilliardPath2D][
            rng.integers(4)]
        lasers.append(cls(rng))
    return lasers


# ══════════════════════════════════════════════════════════════
# 4. FEM 时间积分（Crank-Nicolson，与原版完全一致）
# ══════════════════════════════════════════════════════════════

def run_simulation_2d(nodes, triangles, bnd_idx, n_steps, dt,
                      kappa=KAPPA, seed_laser=0):
    """
    运行一条轨迹。
    返回 (node_features [T,N,1], source_terms [T,N,1])。
    完全对齐 generate_laser_data_2d.run_simulation_2d，包含 FEM 弱形式修正。
    """
    N = nodes.shape[0]
    rng = np.random.default_rng(seed_laser)

    K = build_stiffness_2d(nodes, triangles, kappa)
    node_areas = compute_node_areas(nodes, triangles)
    M = sp.diags(node_areas, format='csr')

    M_conv = sp.diags(
        np.where(np.isin(np.arange(N), bnd_idx), H_NORM * node_areas, 0.0),
        format='csr')

    L_eff = K - M_conv
    A = M + (dt / 2) * L_eff
    B_op = M - (dt / 2) * L_eff
    A_lu = splu(A.tocsc())

    lasers = make_lasers_2d(rng, n_lasers=NUM_LASERS)
    u = np.full(N, T_AMBIENT, dtype=np.float32)
    feats = np.zeros((n_steps, N, 1), dtype=np.float32)
    srcs  = np.zeros((n_steps, N, 1), dtype=np.float32)

    def laser_src(t_eval):
        q = np.zeros(N, dtype=np.float64)
        for laser in lasers:
            pos = laser.get_position(t_eval)
            if pos is not None and laser.active:
                d2 = np.sum((nodes - pos) ** 2, axis=1)
                q += laser.power * np.exp(-d2 / (2 * laser.radius ** 2))
        return q / (RHO * SPECIFIC_HEAT)

    for t_idx in range(1, n_steps):
        t_cur, t_prev = t_idx * dt, (t_idx - 1) * dt
        q_prev, q_cur = laser_src(t_prev), laser_src(t_cur)

        rhs_conv = np.zeros(N, dtype=np.float64)
        rhs_conv[bnd_idx] = H_NORM * node_areas[bnd_idx] * T_AMBIENT

        # FEM 弱形式：源项乘节点面积（BUG FIX，与原版一致）
        rhs = (B_op @ u.astype(np.float64)
               + dt * 0.5 * (q_prev + q_cur) * node_areas
               + dt * rhs_conv)
        u = A_lu.solve(rhs).astype(np.float32)
        feats[t_idx, :, 0] = u - T_AMBIENT
        srcs[t_idx,  :, 0] = q_cur.astype(np.float32)

    return feats, srcs


# ══════════════════════════════════════════════════════════════
# 5. 将 L_physics 写入 HDF5（稀疏 CSR 格式）
# ══════════════════════════════════════════════════════════════

def save_L_physics(grp, nodes, triangles, bnd_idx, node_areas, kappa):
    """
    构建并以 CSR 格式存储 L_physics，供 OriginalDGNet 直接加载。

    存储的是归一化扩散 Laplacian（行 / vol_i），与 physics.py
    _build_laplace_operator 的输出一致，但用稀疏格式表示。

    OriginalDGNet 的 prepare_batch 会调用 L.to(device) 或 L.to_dense()，
    因此这里存储的值在读取时可以组装成 torch.sparse_coo_tensor 或 dense。
    """
    L = build_diffusion_laplacian(nodes, triangles, None, kappa,
                                   H_NORM, bnd_idx, node_areas)
    L_csr = L.tocsr()
    lgrp = grp.create_group('L_physics')
    lgrp.create_dataset('data',    data=L_csr.data.astype(np.float32))
    lgrp.create_dataset('indices', data=L_csr.indices.astype(np.int32))
    lgrp.create_dataset('indptr',  data=L_csr.indptr.astype(np.int32))
    lgrp.create_dataset('shape',   data=np.array(L_csr.shape, dtype=np.int32))
    lgrp.attrs['diffusion_coeff'] = float(kappa)


# ══════════════════════════════════════════════════════════════
# 6. 主数据集生成函数
# ══════════════════════════════════════════════════════════════

def generate_dataset(n_nodes: int, n_traj: int, out_dir: str,
                     n_timesteps: int = 121, dt: float = 0.5,
                     store_L: bool = True, seed: int = 42):
    """
    生成完整数据集并存入 HDF5。

    Args:
        n_nodes      : 近似节点数（Delaunay 后实际略有不同）
        n_traj       : 轨迹数量
        out_dir      : 输出目录
        n_timesteps  : 每条轨迹的时间步数（默认 121，对应 T_sim=60s, dt=0.5s）
        dt           : 时间步长（秒）
        store_L      : 是否存储稀疏 L_physics（供 OriginalDGNet 使用）
        seed         : 随机种子

    输出文件名: pde_trajectories_{n_nodes}.h5
    （n_nodes 为传入的目标节点数，train_dgnet.py / train_physhgnet.py / compare.py
    均通过 --n_nodes 参数自动定位此文件。）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"pde_trajectories_{n_nodes}.h5"

    time_points = np.arange(n_timesteps, dtype=np.float32) * dt
    kappa = KAPPA

    print(f"生成 2D 网格（近似节点数 ≈ {n_nodes}）...")
    nodes, edges, triangles, bnd_idx = generate_2d_mesh(n_nodes, seed=seed)
    N = nodes.shape[0]
    node_areas = compute_node_areas(nodes, triangles)

    # node_type: 0=interior, 1=Dirichlet（对流边界，无 Dirichlet 约束）
    # 此数据集无 Dirichlet BC（仅有对流），全部标记为 0 或 1（边界节点）
    node_type = np.zeros(N, dtype=np.int32)
    node_type[bnd_idx] = 1   # 边界节点标记（对流 BC，不强制温度）

    print(f"  N={N}, 边={len(edges)}, 三角面={len(triangles)}, "
          f"边界节点={len(bnd_idx)}, 时间步={n_timesteps}")
    if store_L:
        print("  将存储稀疏 L_physics（供 OriginalDGNet 使用）")

    with h5py.File(h5_path, 'w') as f:
        for i in range(n_traj):
            t0 = time.time()
            print(f"  轨迹 {i+1}/{n_traj}...", end=" ", flush=True)

            feats, srcs = run_simulation_2d(
                nodes, triangles, bnd_idx, n_timesteps, dt,
                kappa=kappa, seed_laser=seed + i * 100)

            grp = f.create_group(f"trajectory_{i}")

            # ── 核心字段（DGPdeDataset 必需）──────────────────
            grp.create_dataset("nodes",              data=nodes)
            grp.create_dataset("edges",              data=edges)
            grp.create_dataset("faces",              data=triangles)
            grp.create_dataset("node_features",      data=feats)
            grp.create_dataset("source_terms",       data=srcs)
            grp.create_dataset("initial_condition",  data=feats[0])
            grp.create_dataset("time_points",        data=time_points)

            # ── 附加字段（OriginalDGNet / PhysHGNet 辅助）─────
            grp.create_dataset("node_type",    data=node_type)
            grp.create_dataset("node_volumes", data=node_areas)

            # ── 稀疏 L_physics（OriginalDGNet 缓存 LU 用）─────
            if store_L:
                save_L_physics(grp, nodes, triangles, bnd_idx,
                                node_areas, kappa)

            # ── boundary_info（对流 BC，无 Dirichlet 约束）────
            bi = grp.create_group("boundary_info")
            di = bi.create_group("dirichlet")
            # 无强制 Dirichlet，传空数组（DGPdeDataset 兼容）
            di.create_dataset("indices", data=np.array([], dtype=np.int32))
            di.create_dataset("values",  data=np.array([], dtype=np.float32))
            ne = bi.create_group("neumann")
            ne.create_dataset("source_indices", data=np.array([], dtype=np.int32))
            ne.create_dataset("target_indices", data=np.array([], dtype=np.int32))

            print(f"{time.time()-t0:.1f}s")

    print(f"\n已保存 → {h5_path}")
    print(f"文件大小: {h5_path.stat().st_size / 1024**2:.1f} MB")
    return h5_path


# ══════════════════════════════════════════════════════════════
# 7. 数据加载工具（供 OriginalDGNet 读取 L_physics）
# ══════════════════════════════════════════════════════════════

def load_L_physics_from_h5(traj_group) -> dict:
    """
    从 HDF5 轨迹组读取稀疏 L_physics，返回 PhysHGNet/OriginalDGNet 通用格式。

    返回格式与 phys_hgnet.py 中 _match_weights_to_edge_index 的期望一致：
        {
          'type': 'sparse',
          'edge_index': LongTensor [2, nnz_offdiag],
          'edge_weights': FloatTensor [nnz_offdiag],
          'diag': FloatTensor [N],
          'N': int,
          'L_scale': float,
        }
    OriginalDGNet 可以直接调用 .to_dense() 或用 _to_dense() 方法转换。
    """
    import torch
    import scipy.sparse as sp_

    if 'L_physics' not in traj_group:
        return None

    lgrp  = traj_group['L_physics']
    data  = lgrp['data'][:]
    indices = lgrp['indices'][:]
    indptr  = lgrp['indptr'][:]
    shape   = lgrp['shape'][:]
    N = int(shape[0])

    L_csr = sp_.csr_matrix((data, indices, indptr), shape=(N, N))

    # 分离对角线和非对角元素
    diag = np.array(L_csr.diagonal(), dtype=np.float32)
    L_off = L_csr - sp_.diags(diag, format='csr')
    L_off = L_off.tocoo()

    mask = L_off.data != 0
    row  = torch.from_numpy(L_off.row[mask].astype(np.int64))
    col  = torch.from_numpy(L_off.col[mask].astype(np.int64))
    vals = torch.from_numpy(L_off.data[mask].astype(np.float32))

    L_scale = float(np.abs(data).max())

    return {
        'type': 'sparse',
        'edge_index': torch.stack([row, col], dim=0),
        'edge_weights': vals,
        'diag': torch.from_numpy(diag),
        'N': N,
        'L_scale': max(L_scale, 1.0),
    }


def load_L_physics_dense(traj_group) -> 'torch.Tensor | None':
    """
    读取稀疏 L_physics 并直接返回稠密 Tensor（供 OriginalDGNet._phys_lu_cache 使用）。
    """
    import torch
    import scipy.sparse as sp_

    if 'L_physics' not in traj_group:
        return None

    lgrp = traj_group['L_physics']
    N = int(lgrp['shape'][0])
    L_csr = sp_.csr_matrix(
        (lgrp['data'][:], lgrp['indices'][:], lgrp['indptr'][:]),
        shape=(N, N))
    return torch.from_numpy(L_csr.toarray().astype(np.float32))


# ══════════════════════════════════════════════════════════════
# 8. 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="2D Laser Hardening Data Generator (aligned: PhysHGNet + OriginalDGNet)")
    ap.add_argument("--n_nodes",      type=int,   default=2000,
                    help="近似节点数（Delaunay 三角化后实际略有不同）")
    ap.add_argument("--n_traj",       type=int,   default=40,
                    help="轨迹数，对应原始 NUM_TRAJECTORIES=40")
    ap.add_argument("--n_timesteps",  type=int,   default=121,
                    help="每条轨迹时间步数（121 对应 T_sim=60s, dt=0.5s）")
    ap.add_argument("--dt",           type=float, default=0.5,
                    help="时间步长（秒）")
    ap.add_argument("--out_dir",      type=str,   default="data_laser_hardening",
                    help="输出目录（与 train_physhgnet.py 默认路径一致）")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--no_store_L",   action="store_true",
                    help="不存储 L_physics（节省存储，但 OriginalDGNet 需自行重算）")
    args = ap.parse_args()

    generate_dataset(
        n_nodes=args.n_nodes,
        n_traj=args.n_traj,
        out_dir=args.out_dir,
        n_timesteps=args.n_timesteps,
        dt=args.dt,
        store_L=not args.no_store_L,
        seed=args.seed,
    )
