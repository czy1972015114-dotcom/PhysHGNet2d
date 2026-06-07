"""
train_dgnet.py — DGNet baseline training (single-GPU or DDP).

Mirrors train_physhgnet.py so run_all.sh can call both with identical flags:

  # 7-GPU DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
      train_dgnet.py --epochs 15 --n-nodes 2000 --data-dir data_laser_hardening

Logs MSE and RNE (relative L2) every epoch, identical metrics to PhysHGNet.
Checkpoints -> checkpoints/dgnet/best_{n}.pth  (used by the benchmark scripts).
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

from dgnet import DGNet
from dataset import DGPdeDataset, create_dg_loader
from metrics import mse as mse_metric, rne as rne_metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-nodes", type=int, default=2000)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data_laser_hardening")
    p.add_argument("--train-time-steps", type=int, default=7)
    p.add_argument("--exp-name", type=str, default="dgnet")
    return p.parse_args()


def is_ddp():
    return "RANK" in os.environ


def main():
    args = parse_args()
    _ddp = is_ddp()
    if _ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    base_dir = pathlib.Path(__file__).parent.resolve()
    if args.data_path:
        data_path = (pathlib.Path(args.data_path) if pathlib.Path(args.data_path).is_absolute()
                     else base_dir / args.data_path)
    else:
        data_path = base_dir / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"
    ckpt_dir = base_dir / "checkpoints" / args.exp_name

    if rank == 0:
        print("=" * 60)
        print(f"DGNet Training [{args.exp_name}]  GPUs={world_size}  Epochs={args.epochs}")
        print(f"  Data: {data_path}")
        print(f"  Checkpoints: {ckpt_dir}")
        print("=" * 60)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data not found: {data_path}\n"
                f"Run: python generate_laser_data_aligned.py --n_nodes {args.n_nodes} "
                f"--out_dir {args.data_dir}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if _ddp:
        dist.barrier()

    with h5py.File(data_path, "r") as f:
        all_keys = sorted(f.keys())
    split = max(1, int(0.8 * len(all_keys)))
    train_keys, val_keys = all_keys[:split], (all_keys[split:] or [all_keys[-1]])

    train_ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                            rank=rank, trajectory_keys=train_keys)
    val_ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                          rank=rank, trajectory_keys=val_keys)
    tr_sampler = DistributedSampler(train_ds, world_size, rank) if _ddp else None
    va_sampler = DistributedSampler(val_ds, world_size, rank) if _ddp else None
    train_loader = create_dg_loader(train_ds, batch_size=args.batch_size,
                                    shuffle=(tr_sampler is None), num_workers=2, sampler=tr_sampler)
    val_loader = create_dg_loader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2, sampler=va_sampler)

    config = {"spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
              "residual_hidden_dim": 128, "residual_num_layers": 5,
              "n_nodes": args.n_nodes}
    model = DGNet(config).to(device)
    if rank == 0:
        print(f"  DGNet parameters: {model.num_parameters():,}")
    if _ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    def to_device(batch):
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

    def run_epoch(loader, train):
        model.train(train)
        tot_loss = tot_mse = tot_rne = n = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, disable=(rank != 0), desc=(" train" if train else " val ")):
                batch = to_device(batch)
                tgt = batch["node_features"]
                if train:
                    optimizer.zero_grad()
                out = model(batch)["u_final"]
                loss = (torch.nn.functional.mse_loss(out[:, 1], tgt[:, 1]) +
                        torch.nn.functional.mse_loss(out[:, -1], tgt[:, -1]))
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                tot_loss += loss.item(); n += 1
                tot_mse += mse_metric(out[:, -1], tgt[:, -1])
                tot_rne += rne_metric(out[:, -1], tgt[:, -1])
        n = max(n, 1)
        return {"loss": tot_loss / n, "mse": tot_mse / n, "rne": tot_rne / n}

    best = float("inf")
    history = defaultdict(list)
    for epoch in range(args.epochs):
        if _ddp and tr_sampler:
            tr_sampler.set_epoch(epoch)
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        scheduler.step()
        if rank == 0:
            history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train MSE={tr['mse']:.5f} RNE={tr['rne']:.4f} | "
                  f"val MSE={va['mse']:.5f} RNE={va['rne']:.4f} | {time.time()-t0:.1f}s")
            mod = model.module if _ddp else model
            ckpt = {"epoch": epoch, "model_state": mod.state_dict(),
                    "config": config, "val_mse": va["mse"], "val_rne": va["rne"],
                    "history": dict(history)}
            torch.save(ckpt, ckpt_dir / f"last_{args.n_nodes}.pth")
            if va["loss"] < best:
                best = va["loss"]
                torch.save(ckpt, ckpt_dir / f"best_{args.n_nodes}.pth")
                print(f"  -> new best (val_loss={best:.6f})")
    if rank == 0:
        print(f"\nDGNet done. Best val loss {best:.6f}. Checkpoints in {ckpt_dir}")
    if _ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
