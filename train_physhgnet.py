"""
train_physhgnet.py — PhysHGNet training entry point (DDP, 2-GPU).

Usage:
    # Full PhysHGNet (all innovations enabled):
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_physhgnet.py

    # Ablation: disable C1 (physics anchor)
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_physhgnet.py --no-c1

    # Ablation: disable C2 (learned coarse operator)
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_physhgnet.py --no-c2

    # Ablation: disable C3 (dual-scale GNN)
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_physhgnet.py --no-c3

    # Ablation: baseline (all OFF = Structured DGNet)
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_physhgnet.py --no-c1 --no-c2 --no-c3
"""

import os
import argparse
import pathlib
import time
from collections import defaultdict

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import h5py
from tqdm import tqdm

from phys_hgnet import PhysHGNet
from dataset import DGPdeDataset, create_dg_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-c1", action="store_true", help="Disable C1 (physics anchor)")
    p.add_argument("--no-c2", action="store_true", help="Disable C2 (learned coarse op)")
    p.add_argument("--no-c3", action="store_true", help="Disable C3 (dual-scale GNN)")
    p.add_argument("--no-vn", action="store_true", help="Disable virtual nodes (C3 sub-ablation)")
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--m-anchors", type=int, default=64)
    p.add_argument("--coarse-layers", type=int, default=4)
    p.add_argument("--k-vn", type=int, default=4)
    p.add_argument("--n-nodes", type=int, default=2000,
                   help="训练数据节点数，自动定位 pde_trajectories_{n_nodes}.h5")
    p.add_argument("--data-path", type=str, default=None,
                   help="显式指定 HDF5 路径，传入后覆盖自动推断")
    p.add_argument("--data-dir", type=str, default="data_laser_hardening",
                   help="数据目录")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


def is_ddp():
    return "RANK" in os.environ


def get_rank():
    return int(os.environ.get("RANK", 0))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v, n=1):
        self.sum += v * n
        self.n += n

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def rel_err(pred, target):
    with torch.no_grad():
        return (torch.norm(pred - target) /
                torch.norm(target).clamp(min=1e-8)).item()


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, optimizer, device, rank):
    model.train()
    m_loss, m_err = Meter(), Meter()
    bar = tqdm(loader, disable=(rank != 0), desc="  train")
    for batch in bar:
        batch = to_device(batch, device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch["node_features"]
        l1 = torch.nn.functional.mse_loss(out["u_final"][:, 1], target[:, 1])
        lT = torch.nn.functional.mse_loss(out["u_final"][:, -1], target[:, -1])
        loss = l1 + lT
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        m_loss.update(loss.item())
        m_err.update(rel_err(out["u_final"][:, -1], target[:, -1]))
        if rank == 0:
            bar.set_postfix(loss=f"{m_loss.avg:.4f}", relErr=f"{m_err.avg:.4f}")
    return {"loss": m_loss.avg, "rel_err": m_err.avg}


@torch.no_grad()
def eval_one_epoch(model, loader, device, rank):
    model.eval()
    m_loss, m_err = Meter(), Meter()
    for batch in tqdm(loader, disable=(rank != 0), desc="  val  "):
        batch = to_device(batch, device)
        out = model(batch)
        target = batch["node_features"]
        l1 = torch.nn.functional.mse_loss(out["u_final"][:, 1], target[:, 1])
        lT = torch.nn.functional.mse_loss(out["u_final"][:, -1], target[:, -1])
        m_loss.update((l1 + lT).item())
        m_err.update(rel_err(out["u_final"][:, -1], target[:, -1]))
    return {"loss": m_loss.avg, "rel_err": m_err.avg}


def main():
    args = parse_args()
    _ddp = is_ddp()
    if _ddp:
        setup_ddp()

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        parts = ["phys_hgnet"]
        if args.no_c1: parts.append("noC1")
        if args.no_c2: parts.append("noC2")
        if args.no_c3: parts.append("noC3")
        if args.no_vn: parts.append("noVN")
        exp_name = "_".join(parts)

    base_dir = pathlib.Path(__file__).parent.resolve()
    ckpt_dir = base_dir / "checkpoints" / exp_name
    # --data-path 显式传入时直接使用，否则按 --n-nodes 自动构造
    if args.data_path:
        data_path = (pathlib.Path(args.data_path) if pathlib.Path(args.data_path).is_absolute()
                    else base_dir / args.data_path)
    else:
        data_path = base_dir / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"

    if rank == 0:
        print("=" * 60)
        print(f"PhysHGNet Training  [{exp_name}]")
        print(f"  GPUs: {world_size}  |  Epochs: {args.epochs}")
        print(f"  Data: {data_path}")
        print(f"  Checkpoints: {ckpt_dir}")
        print("=" * 60)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Run: python generate_laser_data_aligned.py --n_nodes {args.n_nodes} --out_dir {args.data_dir}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _ddp:
        dist.barrier()

    # Dataset
    with h5py.File(data_path, "r") as f:
        all_keys = sorted(f.keys())
    split_idx = max(1, int(0.8 * len(all_keys)))
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:] or [train_keys[-1]]

    if rank == 0:
        print(f"  Train trajectories: {len(train_keys)} | Val: {len(val_keys)}")

    train_ds = DGPdeDataset(data_path, train_time_steps=7, rank=rank, trajectory_keys=train_keys)
    val_ds = DGPdeDataset(data_path, train_time_steps=7, rank=rank, trajectory_keys=val_keys)

    if _ddp:
        tr_sampler = DistributedSampler(train_ds, world_size, rank)
        va_sampler = DistributedSampler(val_ds, world_size, rank)
    else:
        tr_sampler = va_sampler = None

    train_loader = create_dg_loader(train_ds, batch_size=args.batch_size,
                                    shuffle=(tr_sampler is None), num_workers=2,
                                    sampler=tr_sampler)
    val_loader = create_dg_loader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2, sampler=va_sampler)

    # Model
    config = {
        "spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
        "operator_type": "laplace",
        "operator_hidden_dim": 64, "operator_num_layers": 3,
        "residual_hidden_dim": 128, "residual_num_layers": 5,
        "coarse_num_layers": args.coarse_layers,
        "k_virtual_nodes": args.k_vn,
        "m_anchors": args.m_anchors,
        "q_local": 4, "k_coarse": 6,
        "cg_max_iter": 50, "cg_tol": 1e-6,
        "use_physics_anchor": not args.no_c1,
        "use_learned_coarse": not args.no_c2,
        "use_dual_scale_gnn": not args.no_c3,
        "use_virtual_nodes": not args.no_vn,
    }

    model = PhysHGNet(config).to(device)

    if rank == 0:
        print(model.ablation_summary())
        print(f"  Parameters: {model.num_parameters():,}")

    if _ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer: higher LR for learnable weights (α params)
    alpha_params = [p for n, p in model.named_parameters()
                    if "raw_alpha" in n or "raw_lambda" in n]
    other_params = [p for n, p in model.named_parameters()
                    if "raw_alpha" not in n and "raw_lambda" not in n]
    optimizer = optim.Adam([
        {"params": other_params, "lr": args.lr},
        {"params": alpha_params, "lr": args.lr * 10},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    best_val_loss = float("inf")
    history = defaultdict(list)

    for epoch in range(args.epochs):
        if _ddp and tr_sampler:
            tr_sampler.set_epoch(epoch)

        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, rank)
        va = eval_one_epoch(model, val_loader, device, rank)
        scheduler.step()

        if rank == 0:
            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(va["loss"])
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train_loss={tr['loss']:.5f} relErr={tr['rel_err']:.4f} | "
                  f"val_loss={va['loss']:.5f} relErr={va['rel_err']:.4f} | "
                  f"{elapsed:.1f}s")

            _module = model.module if _ddp else model
            ckpt = {
                "epoch": epoch, "model_state": _module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "ablation": {"C1": not args.no_c1, "C2": not args.no_c2,
                             "C3": not args.no_c3, "VN": not args.no_vn},
                "val_loss": va["loss"],
                "history": dict(history),
            }
            n_nodes = args.n_nodes
            torch.save(ckpt, ckpt_dir / f"last_{n_nodes}.pth")
            if va["loss"] < best_val_loss:
                best_val_loss = va["loss"]
                torch.save(ckpt, ckpt_dir / f"best_{n_nodes}.pth")
                print(f"  → New best model (val_loss={best_val_loss:.6f})")

    if rank == 0:
        print(f"\nPhysHGNet training complete. Best val loss: {best_val_loss:.6f}")
        print(f"Checkpoints saved in: {ckpt_dir}")

    if _ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()

