"""
train_dgnet.py — DGNet 训练入口（DDP，多卡）

与仓库 czy1972015114-dotcom/DGNEt 的 train_structured.py + heat_equation_benchmark.py
对齐，主要变化：
  [1] 调度器改为 warmup（3 epoch 线性）+ cosine decay，与仓库 lr_lambda 一致。
  [2] 新增 --loss_last_only / --grad_checkpoint 开关（与仓库 SimpleLoss 对齐）。
  [3] 优化器对 alpha 类参数使用 10× 学习率（与仓库 train_model 一致）。
  [4] 打印格式对齐仓库输出（Ep | Loss | RelErr | Time | GPU MB）。

Usage:
    # 单卡
    python train_dgnet.py

    # 双卡 DDP
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_dgnet.py

    # 仅训练最后一步（节省显存）
    torchrun --nproc_per_node=2 train_dgnet.py --loss_last_only --grad_checkpoint
"""

import os
import math
import time
import pathlib
import argparse

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import h5py

from dgnet import DGNet, DGNetLoss, DGNetTrainer
from dataset import DGPdeDataset, create_dg_loader


# ─────────────────────────────────────────────────────────────
# DDP 初始化
# ─────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def cleanup_ddp():
    dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ─────────────────────────────────────────────────────────────
# 学习率调度（对齐仓库 lr_lambda：warmup + cosine decay）
# ─────────────────────────────────────────────────────────────

def make_lr_lambda(warmup_epochs: int, total_epochs: int):
    """
    返回与仓库 heat_equation_benchmark.py lr_lambda 完全一致的调度函数：
      epoch < warmup_epochs  → 线性 warmup：(epoch+1) / warmup_epochs
      epoch >= warmup_epochs → cosine decay：0.5 * (1 + cos(π · progress))
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main():
    # ── 命令行参数 ───────────────────────────────────────────
    ap = argparse.ArgumentParser(description='DGNet Training')
    ap.add_argument('--data_dir',  type=str, default='data_laser_hardening',
                    help='数据目录（相对于脚本所在位置）')
    ap.add_argument('--n_nodes',   type=int, default=2000,
                    help='训练数据节点数，自动定位 pde_trajectories_{n_nodes}.h5')
    ap.add_argument('--data_path', type=str, default=None,
                    help='显式指定 HDF5 路径，传入后覆盖 --data_dir + --n_nodes')
    ap.add_argument('--ckpt_dir',  type=str, default='checkpoints/dgnet')
    ap.add_argument('--batch_size',     type=int,   default=2)
    ap.add_argument('--num_workers',    type=int,   default=2)
    ap.add_argument('--train_time_steps', type=int, default=7)
    ap.add_argument('--num_epochs',     type=int,   default=15)
    ap.add_argument('--lr',             type=float, default=5e-4)
    ap.add_argument('--warmup_epochs',  type=int,   default=3,
                    help='线性 warmup epoch 数（之后 cosine decay）')
    ap.add_argument('--operator_hidden_dim', type=int, default=64)
    ap.add_argument('--operator_num_layers', type=int, default=3)
    ap.add_argument('--residual_hidden_dim', type=int, default=128)
    ap.add_argument('--residual_num_layers', type=int, default=5)
    ap.add_argument('--gradient_clip',  type=float, default=1.0)
    ap.add_argument('--loss_last_only', action='store_true',
                    help='只计算最后一步 loss（节省显存，与仓库 --loss_last_only 一致）')
    ap.add_argument('--grad_checkpoint', action='store_true',
                    help='开启梯度检查点（节省显存，与仓库 --grad_checkpoint 一致）')
    args = ap.parse_args()

    # ── DDP 初始化 ───────────────────────────────────────────
    setup_ddp()
    rank       = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    base_dir  = pathlib.Path(__file__).parent.resolve()
    # --data_path 显式传入时直接使用，否则按 --n_nodes 自动构造
    if args.data_path:
        data_path = pathlib.Path(args.data_path) if pathlib.Path(args.data_path).is_absolute() \
                    else base_dir / args.data_path
    else:
        data_path = base_dir / args.data_dir / f'pde_trajectories_{args.n_nodes}.h5'
    ckpt_dir  = base_dir / args.ckpt_dir

    config = {
        'data_path':             str(data_path),
        'batch_size':            args.batch_size,
        'num_workers':           args.num_workers,
        'train_time_steps':      args.train_time_steps,
        'spatial_dim':           2,
        'feature_dim':           1,
        'output_dim':            1,
        'operator_type':         'laplace',
        'operator_hidden_dim':   args.operator_hidden_dim,
        'operator_num_layers':   args.operator_num_layers,
        'residual_hidden_dim':   args.residual_hidden_dim,
        'residual_num_layers':   args.residual_num_layers,
        'num_epochs':            args.num_epochs,
        'learning_rate':         args.lr,
        'warmup_epochs':         args.warmup_epochs,
        'gradient_clip':         args.gradient_clip,
        'loss_type':             'mse',
        'loss_last_only':        args.loss_last_only,
        'use_checkpoint':        args.grad_checkpoint,
        'checkpoint_dir':        str(ckpt_dir),
        'rank':                  rank,
        'n_nodes':               args.n_nodes,
    }

    if is_main():
        print('=' * 60)
        print('DGNet Training  (Discrete Green Network)')
        print(f'  GPUs       : {world_size}')
        print(f'  Epochs     : {args.num_epochs}  (warmup={args.warmup_epochs})')
        print(f'  LR         : {args.lr}')
        print(f'  Loss last  : {args.loss_last_only}')
        print(f'  Grad ckpt  : {args.grad_checkpoint}')
        print(f'  Data       : {data_path}')
        print(f'  Checkpoints: {ckpt_dir}')
        print('=' * 60)

        if not data_path.exists():
            raise FileNotFoundError(
                f'Data not found: {data_path}\n'
                f'Run: python generate_laser_data_aligned.py --out_dir {args.data_dir}')
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    # ── 数据集 ───────────────────────────────────────────────
    with h5py.File(data_path, 'r') as f:
        all_keys = sorted(list(f.keys()))

    split_idx  = max(1, int(0.8 * len(all_keys)))
    train_keys = all_keys[:split_idx]
    val_keys   = all_keys[split_idx:] or [train_keys[-1]]

    if is_main():
        print(f'  Train trajectories: {len(train_keys)} | Val: {len(val_keys)}')

    train_ds = DGPdeDataset(data_path,
                            train_time_steps=config['train_time_steps'],
                            rank=rank, trajectory_keys=train_keys)
    val_ds   = DGPdeDataset(data_path,
                            train_time_steps=config['train_time_steps'],
                            rank=rank, trajectory_keys=val_keys)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = create_dg_loader(train_ds, batch_size=config['batch_size'],
                                    shuffle=False, num_workers=config['num_workers'],
                                    sampler=train_sampler)
    val_loader   = create_dg_loader(val_ds,   batch_size=config['batch_size'],
                                    shuffle=False, num_workers=config['num_workers'],
                                    sampler=val_sampler)

    # ── 模型 ─────────────────────────────────────────────────
    model = DGNet(config)
    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  DGNet parameters: {n_params:,}')

    # ── 优化器（对齐仓库：alpha 类参数 10× 学习率）─────────
    # DGNet 本身没有 raw_alpha_loc，但保留此模式与 PhysHGNet 统一
    alpha_ids = {id(p) for name, p in model.named_parameters()
                 if 'raw_alpha' in name}
    alpha_p = [p for p in model.parameters() if id(p) in alpha_ids]
    other_p = [p for p in model.parameters() if id(p) not in alpha_ids]

    if alpha_p:
        optimizer = optim.Adam([
            {'params': other_p, 'lr': args.lr},
            {'params': alpha_p, 'lr': args.lr * 10},
        ], betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), eps=1e-8)

    # ── 调度器（warmup + cosine，对齐仓库 lr_lambda）────────
    lr_lambda = make_lr_lambda(args.warmup_epochs, args.num_epochs)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = DGNetLoss(config)

    # ── 训练 ─────────────────────────────────────────────────
    trainer = DGNetTrainer(model, optimizer, loss_fn, config,
                           rank=rank, local_rank=local_rank,
                           scheduler=scheduler)
    trainer.train(train_loader, val_loader, config['num_epochs'])

    if is_main():
        print('\nDGNet training complete.')
        print(f'Best val loss : {trainer.best_val_loss:.6f}')
        print(f'Checkpoints   : {ckpt_dir}')

    cleanup_ddp()


if __name__ == '__main__':
    main()

