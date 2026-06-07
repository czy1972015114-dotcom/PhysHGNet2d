"""
train_physhgnet.py — PhysHGNet training entry point (single-GPU or DDP).
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
from metrics import mse as mse_metric, rne as rne_metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-c1", action="store_true")
    p.add_argument("--no-mg", action="store_true")
    p.add_argument("--no-c2", action="store_true")
    p.add_argument("--no-c3", action="store_true")
    p.add_argument("--no-vn", action="store_true")
    p.add_argument("--static-anchors", action="store_true")
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--m-anchors", type=int, default=64)
    p.add_argument("--coarse-layers", type=int, default=4)
    p.add_argument("--k-vn", type=int, default=4)
    p.add_argument("--n-nodes", type=int, default=2000)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data_laser_hardening")
    p.add_argument("--train-time-steps", type=int, default=7)
    return p.parse_args()


def is_ddp():
    return "RANK" in os.environ


def main():
    args = parse_args()
    _ddp = is_ddp()
    if _ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.exp_name:
        exp_name = args.exp_name
    else:
        parts = ["phys_hgnet"]
        for flag, tag in [(args.no_c1, "noC1"), (args.no_mg, "noMG"),
                          (args.no_c2, "noC2"), (args.no_c3, "noC3"),
                          (args.no_vn, "noVN"), (args.static_anchors, "static")]:
            if flag:
                parts.append(tag)
        exp_name = "_".join(parts)

    base_dir = pathlib.Path(__file__).parent.resolve()
    if args.data_path:
        data_path = (pathlib.Path(args.data_path)
                     if pathlib.Path(args.data_path).is_absolute()
                     else base_dir / args.data_path)
    else:
        data_path = base_dir / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"
    ckpt_dir = base_dir / "checkpoints" / exp_name

    if rank == 0:
        print("=" * 60)
        print(f"PhysHGNet Training [{exp_name}]  GPUs={world_size}  Epochs={args.epochs}")
        print(f"  Data: {data_path}")
        print("=" * 60)
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if _ddp:
        dist.barrier()

    with h5py.File(data_path, "r") as f:
        all_keys = sorted(f.keys())
    split = max(1, int(0.8 * len(all_keys)))
    train_keys = all_keys[:split]
    val_keys   = all_keys[split:] or [all_keys[-1]]

    train_ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                            rank=rank, trajectory_keys=train_keys)
    val_ds   = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                            rank=rank, trajectory_keys=val_keys)
    tr_sampler = DistributedSampler(train_ds, world_size, rank) if _ddp else None
    va_sampler = DistributedSampler(val_ds,   world_size, rank) if _ddp else None
    train_loader = create_dg_loader(train_ds, batch_size=args.batch_size,
                                    shuffle=(tr_sampler is None),
                                    num_workers=2, sampler=tr_sampler)
    val_loader   = create_dg_loader(val_ds,   batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=2, sampler=va_sampler)

    config = {
        "spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
        "operator_type": "laplace",
        "operator_hidden_dim": 64, "operator_num_layers": 3,
        "residual_hidden_dim": 128, "residual_num_layers": 5,
        "coarse_num_layers": args.coarse_layers,
        "k_virtual_nodes": args.k_vn,
        "m_anchors": args.m_anchors, "q_local": 4, "k_coarse": 6,
        "cg_max_iter": 50, "cg_tol": 1e-6,
        "use_physics_anchor":  not args.no_c1,
        "use_mg_precond":      not args.no_mg,
        "use_learned_coarse":  not args.no_c2,
        "use_dual_scale_gnn":  not args.no_c3,
        "use_virtual_nodes":   not args.no_vn,
        "dynamic_anchors":     not args.static_anchors,
        "anchor_temp_loss_weight": 0.1,   # C1 辅助损失权重
    }
    model = PhysHGNet(config).to(device)
    if rank == 0:
        print(model.ablation_summary())
    if _ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    alpha_params = [p for n, p in model.named_parameters()
                    if "raw_alpha" in n or "raw_weights" in n]
    other_params = [p for n, p in model.named_parameters()
                    if "raw_alpha" not in n and "raw_weights" not in n]
    optimizer = optim.Adam([
        {"params": other_params, "lr": args.lr},
        {"params": alpha_params, "lr": args.lr * 10},
    ])
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
        tot_loss = tot_mse = tot_rne = tot_it = n = 0.0
        ctx  = torch.enable_grad() if train else torch.no_grad()
        core = model.module if _ddp else model
        with ctx:
            for batch in tqdm(loader, disable=(rank != 0),
                              desc=(" train" if train else " val  ")):
                batch = to_device(batch)
                tgt   = batch["node_features"]
                if train:
                    optimizer.zero_grad()
                out  = model(batch)["u_final"]
                loss = (torch.nn.functional.mse_loss(out[:, 1],  tgt[:, 1]) +
                        torch.nn.functional.mse_loss(out[:, -1], tgt[:, -1]))
                if train:
                    # ── C1 锚点温度辅助损失（训练 encoder 选热节点）─────────
                    # 梯度路径: L_anchor → P → encoder/anchor_protos/gamma_temp
                    if hasattr(core, 'anchor_aux_loss'):
                        loss = loss + core.anchor_aux_loss()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                tot_loss += loss.item()
                n        += 1
                tot_mse  += mse_metric(out[:, -1], tgt[:, -1])
                tot_rne  += rne_metric(out[:, -1], tgt[:, -1])
                tot_it   += core.avg_cg_iters()
        n = max(n, 1)
        return {"loss": tot_loss / n, "mse": tot_mse / n,
                "rne": tot_rne / n,  "cg":  tot_it  / n}

    best    = float("inf")
    history = defaultdict(list)
    for epoch in range(args.epochs):
        if _ddp and tr_sampler:
            tr_sampler.set_epoch(epoch)
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader,   False)
        scheduler.step()
        if rank == 0:
            history["val_loss"].append(va["loss"])
            core = model.module if _ddp else model
            print(f"  [{exp_name}] Epoch {epoch+1:3d}/{args.epochs} "
                  f"| train_loss={tr['loss']:.5f} "
                  f"| val MSE={va['mse']:.5f} RNE={va['rne']:.4f} "
                  f"| CG~{va['cg']:.1f} "
                  f"| {time.time()-t0:.1f}s")
            ckpt = {"epoch": epoch, "model_state": core.state_dict(),
                    "config": config, "val_mse": va["mse"],
                    "val_rne": va["rne"], "history": dict(history)}
            torch.save(ckpt, ckpt_dir / f"last_{args.n_nodes}.pth")
            if va["loss"] < best:
                best = va["loss"]
                torch.save(ckpt, ckpt_dir / f"best_{args.n_nodes}.pth")
                print(f"    -> new best (val_loss={best:.6f})")
    if rank == 0:
        print(f"\nDone. Best val loss {best:.6f}. Checkpoints in {ckpt_dir}")
    if _ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
