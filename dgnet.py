"""
dgnet.py — DGNet 模型、损失函数与训练工具

与仓库 czy1972015114-dotcom/DGNEt 的 heat_equation_benchmark.py 对齐，
实现了以下三项修复：

  [BUG FIX 1] LU 缓存：A_phys 的 LU 分解只计算一次，按 (N, device_id, dt) 缓存。
              原版每次 forward 都重新分解 N×N 矩阵，N=5000 时需要数十秒/batch，
              这是推理耗时 131s 的根本原因。

  [BUG FIX 2] dL 显式处理：OperatorCorrector 的输出 ΔL（_corr_scale=1e-5）
              以显式方式加到 RHS，不参与 LU 分解。
              推导（CN 分裂，ΔL 显式）：
                (I - dt/2 * L_phys) u^{n+1}
                = (I + dt/2 * L_phys) u^n
                + dt * ΔL * u^n          ← 显式
                + dt/2*(f_t + f_{t+1})
                + dt * r_uk

  [BUG FIX 3] MPNN 批量化：用 PyG 图拼接（models.batch_graphs）将 B 个图
              合并成一个大图，一次 forward 完成所有样本，消除 Python for 循环。

其他变化：
  - 移除 CuPy 依赖，改用纯 PyTorch torch.linalg.lu_factor/lu_solve。
  - 删除 LUFactorizedSolver autograd Function。
  - DGNetTrainer 保持不变（供 train_dgnet.py 使用）。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional
import os
from collections import defaultdict
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from physics import build_operator, apply_bcs_to_state
from models import OperatorCorrector, NonlinearDynamicsSolver, ResidualSolver


# ─────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def compute_state_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return (torch.norm(pred - target) /
                torch.norm(target).clamp(min=1e-8)).item()


def _to_dense(L: torch.Tensor, device: torch.device) -> torch.Tensor:
    """将 L_physics（稀疏或稠密）转为稠密并移到 device。"""
    if L.is_sparse:
        return L.to_dense().to(device)
    return L.to(device)


# ─────────────────────────────────────────────────────────────
# DGNet 模型
# ─────────────────────────────────────────────────────────────

class DGNet(nn.Module):
    """
    Discrete Green Network，IMEX 时间步进。

    与仓库对齐的三项修复均在 forward 中实现。
    接口（batch 字段）与 PhysHGNet 完全一致。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config      = config
        self.spatial_dim = config['spatial_dim']
        self.feature_dim = config['feature_dim']
        self.output_dim  = config['output_dim']
        self.operator_type = config.get('operator_type', 'laplace')

        hd = config.get('operator_hidden_dim', 64)
        nl = config.get('operator_num_layers',  3)
        rd = config.get('residual_hidden_dim', 128)
        rn = config.get('residual_num_layers',   5)

        self.operator_corrector = OperatorCorrector(
            spatial_dim=self.spatial_dim, hidden_dim=hd, num_layers=nl)
        self.nonlinear_solver = NonlinearDynamicsSolver(
            spatial_dim=self.spatial_dim, node_feature_dim=self.feature_dim,
            output_dim=self.output_dim, hidden_dim=rd, num_processing_layers=rn)
        self.residual_solver = ResidualSolver(
            spatial_dim=self.spatial_dim, node_feature_dim=self.feature_dim,
            output_dim=self.output_dim, hidden_dim=rd, num_processing_layers=rn)

        # [BUG FIX 1] LU 缓存：键 = (N, device_id, dt_rounded)
        # 值 = (lu_factor, pivots, B_phys_dense)
        self._phys_lu_cache: dict = {}

    # ── 构建双向边索引 ────────────────────────────────────────
    @staticmethod
    def _make_bi_edge(edges: torch.Tensor) -> torch.Tensor:
        """edges [E, 2] → bi_e [2, 2E]"""
        return torch.cat([edges, edges.flip(1)], dim=0).T

    # ── [BUG FIX 1+2] 获取/缓存物理求解算子 ─────────────────
    def _get_phys_operators(self, N: int, dt: float, device: torch.device,
                             Lp_raw: torch.Tensor):
        """
        返回 (lu, piv, B_phys)，对相同 (N, device, dt) 只计算一次。

        A_phys = I - dt/2 * L_phys  （仅用 L_phys，不含 ΔL）
        B_phys = I + dt/2 * L_phys

        ΔL 在 forward 中显式加到 RHS，不参与 LU 分解（Bug Fix 2）。
        """
        dev_id   = device.index if device.index is not None else -1
        key      = (N, dev_id, round(dt, 10))
        if key not in self._phys_lu_cache:
            Lp = _to_dense(Lp_raw, device)
            I  = torch.eye(N, device=device, dtype=Lp.dtype)
            A  = (I - (dt / 2) * Lp).double().contiguous()
            B  = (I + (dt / 2) * Lp).float()
            lu, piv = torch.linalg.lu_factor(A)
            self._phys_lu_cache[key] = (lu, piv, B)
            del A, I, Lp          # 释放大矩阵
        return self._phys_lu_cache[key]

    # ── Forward ─────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any],
                use_physics_path:    bool = True,
                use_physics_operator: bool = True,
                use_nn_correction:   bool = True) -> Dict[str, torch.Tensor]:

        nodes         = batch['nodes']
        edges         = batch['edges']
        faces         = batch.get('faces')
        node_volumes  = batch['node_volumes']
        initial_cond  = batch['initial_conditions']   # [B, 1, N, C] 或 [B, N, C]
        source_terms  = batch['source_terms']          # [B, T, N, C]
        time_points   = batch['time_points']
        node_type     = batch.get('node_type',
                                   torch.zeros(nodes.shape[0], dtype=torch.long,
                                               device=nodes.device))
        boundary_info = batch.get('boundary_info', {})
        Lp_raw        = batch.get('L_physics', None)

        # initial_conditions 可能是 [B,1,N,C] 或 [B,N,C]
        u_current = (initial_cond[:, 0] if initial_cond.dim() == 4
                     else initial_cond)                              # [B, N, C]

        B, T, N, C = source_terms.shape
        device     = nodes.device
        dt         = float(time_points[1] - time_points[0]) if T > 1 else 0.01

        # ── 双向边索引（一次性构建，所有步复用）──────────────
        bi_e = self._make_bi_edge(edges)   # [2, 2E]

        # ── [BUG FIX 2] ΔL：只算一次，显式加到 RHS ──────────
        if use_nn_correction:
            delta_L = self.operator_corrector({
                'nodes': nodes, 'edges': edges,
                'node_volumes': node_volumes, 'node_type': node_type})
        else:
            delta_L = None

        # ── 构建 L_physics ───────────────────────────────────
        if use_physics_operator:
            if Lp_raw is None:
                Lp_raw = build_operator(nodes=nodes, edges=edges, faces=faces,
                                        node_volumes=node_volumes,
                                        operator_type=self.operator_type)
        else:
            Lp_raw = torch.zeros(N, N, device=device)

        # ── [BUG FIX 1] 获取缓存的 LU 分解 ──────────────────
        lu, piv, B_phys = self._get_phys_operators(N, dt, device, Lp_raw)

        # ── 时间步进 ──────────────────────────────────────────
        u_hist    = torch.zeros(B, T, N, C, device=device)
        u_hist[:, 0] = u_current

        for t in range(T - 1):
            f_cur  = source_terms[:, t]       # [B, N, C]
            f_next = source_terms[:, t + 1]

            if use_physics_path:
                # [BUG FIX 3] 批量非线性项（PyG 图拼接，消除 for 循环）
                r_uk = self.nonlinear_solver.forward_batched(
                    nodes, bi_e, u_current, node_type)   # [B, N, C]

                # RHS（CN + 显式 ΔL + source + 非线性）
                # B_phys @ u: [N,N] × [N,B·C] → [N,B·C] → [B,N,C]
                u2d      = u_current.permute(1, 0, 2).reshape(N, B * C)
                Bu       = (B_phys @ u2d).reshape(N, B, C).permute(1, 0, 2)

                # [BUG FIX 2] ΔL 显式：dt * ΔL @ u
                if delta_L is not None:
                    dLu = (delta_L @ u2d).reshape(N, B, C).permute(1, 0, 2)
                    dLu_term = dt * dLu
                else:
                    dLu_term = torch.zeros_like(u_current)

                rhs = (Bu
                       + dLu_term
                       + (dt / 2) * (f_cur + f_next)
                       + dt * r_uk)                              # [B, N, C]

                # [BUG FIX 1] 批量 lu_solve：[N, B·C]
                rhs2d  = rhs.permute(1, 0, 2).reshape(N, B * C)
                sol2d  = torch.linalg.lu_solve(lu, piv, rhs2d.double()).float()
                u_phys = sol2d.reshape(N, B, C).permute(1, 0, 2)   # [B, N, C]
            else:
                u_phys = torch.zeros_like(u_current)

            # [BUG FIX 3] 批量残差修正
            if use_nn_correction:
                u_net = self.residual_solver.forward_batched(
                    nodes, bi_e, u_current, node_type, boundary_info)  # [B, N, C]
            else:
                u_net = torch.zeros_like(u_current)

            u_next = u_phys + u_net

            # 边界条件
            if boundary_info and isinstance(boundary_info, dict) \
                    and 'dirichlet' in boundary_info:
                di  = boundary_info['dirichlet']
                idx = di.get('indices')
                if idx is not None and len(idx) > 0:
                    val = di.get('values')
                    idx = idx.to(device)
                    if val is not None:
                        val = val.to(device)
                        for c in range(C):
                            u_next[:, idx, c] = (val.unsqueeze(0)
                                                  if val.dim() == 1 else val)

            u_hist[:, t + 1] = u_next
            u_current = u_next.detach()

        return {'u_final': u_hist}


# ─────────────────────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────────────────────

class DGNetLoss(nn.Module):
    """
    与仓库 SimpleLoss 对齐：
      loss_last_only=False → L(step1) + L(stepT)
      loss_last_only=True  → L(stepT) 只算最后一步
    """

    def __init__(self, config: Dict):
        super().__init__()
        loss_type = config.get('loss_type', 'mse')
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        self.loss_last_only = config.get('loss_last_only', False)

    def forward(self, predictions: Dict, targets: torch.Tensor) -> Dict:
        u  = predictions['u_final']
        lT = self.base_loss(u[:, -1], targets[:, -1])
        if self.loss_last_only:
            return {'total_loss': lT,
                    'first_step_loss': torch.tensor(0.0),
                    'final_step_loss': lT}
        l1 = self.base_loss(u[:, 1], targets[:, 1])
        return {'total_loss': l1 + lT,
                'first_step_loss': l1,
                'final_step_loss': lT}


# ─────────────────────────────────────────────────────────────
# 训练器（DDP 感知，接口不变）
# ─────────────────────────────────────────────────────────────

class DGNetTrainer:
    """DGNet 训练/验证循环（DDP 感知）。"""

    def __init__(self, model, optimizer, loss_fn, config,
                 rank: int, local_rank: int, scheduler=None):
        self.model_device = torch.device(f'cuda:{local_rank}')
        self.rank         = rank
        self.local_rank   = local_rank
        self.config       = config

        model.to(self.model_device)
        self.model     = DDP(model, device_ids=[self.local_rank],
                              find_unused_parameters=False)
        self.optimizer = optimizer
        self.loss_fn   = loss_fn
        self.scheduler = scheduler

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = defaultdict(list)
        self.val_history   = defaultdict(list)

        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/dgnet')
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader, num_epochs: int):
        if self.rank == 0:
            print(f"[DGNet] Training {num_epochs} epochs on "
                  f"{dist.get_world_size()} GPU(s).")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_m = self._train_epoch(train_loader)
            val_m   = self._evaluate_epoch(val_loader)

            if self.rank == 0:
                for k, v in train_m.items():
                    self.train_history[k].append(v)
                for k, v in val_m.items():
                    self.val_history[k].append(v)
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"train_loss={train_m['loss']:.5f} "
                      f"relErr={train_m['relative_error']:.4f} | "
                      f"val_loss={val_m['loss']:.5f} "
                      f"relErr={val_m['relative_error']:.4f}")

                if val_m['loss'] < self.best_val_loss:
                    self.best_val_loss = val_m['loss']
                    n = self.config.get('n_nodes', '')
                    suffix = f'_{n}' if n else ''
                    self.save_checkpoint(f'best{suffix}.pth')
                    print("  → New best model saved.")
                self.save_checkpoint(f'last{suffix}.pth')

            if self.scheduler:
                self.scheduler.step()

    def _train_epoch(self, loader) -> Dict:
        self.model.train()
        loss_m = AverageMeter()
        err_m  = AverageMeter()
        bar    = tqdm(loader, desc=f"train", disable=(self.rank != 0),
                      leave=False)
        for batch in bar:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            tgt  = batch['node_features']
            ld   = self.loss_fn(pred, tgt)
            ld['total_loss'].backward()
            if self.config.get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['gradient_clip'])
            self.optimizer.step()
            loss_m.update(ld['total_loss'].item())
            with torch.no_grad():
                err_m.update(
                    compute_state_error(pred['u_final'][:, -1], tgt[:, -1]))
            if self.rank == 0:
                bar.set_postfix(loss=f"{loss_m.avg:.4f}",
                                relErr=f"{err_m.avg:.4f}")

        return self._reduce({'loss': loss_m.avg, 'relative_error': err_m.avg})

    def _evaluate_epoch(self, loader) -> Dict:
        self.model.eval()
        loss_m = AverageMeter()
        err_m  = AverageMeter()
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                pred  = self.model(batch)
                tgt   = batch['node_features']
                ld    = self.loss_fn(pred, tgt)
                loss_m.update(ld['total_loss'].item())
                err_m.update(
                    compute_state_error(pred['u_final'][:, -1], tgt[:, -1]))
        return self._reduce({'loss': loss_m.avg, 'relative_error': err_m.avg})

    def _reduce(self, metrics: Dict) -> Dict:
        t = torch.tensor(list(metrics.values()),
                         device=self.model_device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
        return {k: v.item() for k, v in zip(metrics.keys(), t)}

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.model_device)
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._to_device(v) for v in data)
        return data

    def save_checkpoint(self, filename: str):
        if self.rank != 0:
            return
        ckpt = {
            'epoch':              self.current_epoch,
            'model_state_dict':   self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss':      self.best_val_loss,
            'config':             self.config,
            'train_history':      dict(self.train_history),
            'val_history':        dict(self.val_history),
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(ckpt, path)
        if self.rank == 0:
            print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, filename: str):
        path       = os.path.join(self.checkpoint_dir, filename)
        map_loc    = {f'cuda:0': f'cuda:{self.local_rank}'}
        ckpt       = torch.load(path, map_location=map_loc)
        self.model.module.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.best_val_loss = ckpt['best_val_loss']
        if self.rank == 0:
            print(f"Loaded checkpoint: {path} (epoch {self.current_epoch})")
