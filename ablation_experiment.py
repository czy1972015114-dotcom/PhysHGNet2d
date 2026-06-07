"""
ablation_experiment.py — PhysHGNet 组件消融（支持 7 卡 DDP）

用 torchrun 启动即可全程多卡：
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \\
      ablation_experiment.py --n-nodes 2000 --data-dir data_laser_hardening \\
      --epochs 15 --configs full no_c1 static_anchor no_mg no_c2 no_c3 no_vn baseline

各 config 串行训练（每个 config 内部使用所有卡并行），rank 0 最终打印对比表。
"""
import os
import csv
import argparse
import pathlib

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phys_hgnet import PhysHGNet
from dataset import DGPdeDataset, create_dg_loader
from metrics import mse as mse_metric, rne as rne_metric


# ── 消融配置 ──────────────────────────────────────────────────────────────────
def cfg_for(name, base):
    c = dict(base)
    c.update({
        "use_physics_anchor": True, "use_mg_precond": True,
        "use_learned_coarse": True, "use_dual_scale_gnn": True,
        "use_virtual_nodes": True,  "dynamic_anchors": True,
    })
    if   name == "full":           pass
    elif name == "no_c1":          c["use_physics_anchor"] = False
    elif name == "static_anchor":  c["dynamic_anchors"] = False
    elif name == "no_mg":          c["use_mg_precond"] = False
    elif name == "no_c2":          c["use_learned_coarse"] = False
    elif name == "no_c3":          c["use_dual_scale_gnn"] = False
    elif name == "no_vn":          c["use_virtual_nodes"] = False
    elif name == "baseline":
        # baseline = 无任何神经修正，仅保留精细尺度 CG 物理求解
        # 等价于: 均匀锚点 + Jacobi CG + 无 C2/C3 + 静态锚点
        c.update({"use_physics_anchor": False, "use_mg_precond": False,
                  "use_learned_coarse": False, "use_dual_scale_gnn": False,
                  "use_virtual_nodes": False,  "dynamic_anchors": False})
    elif name == "pure_physics":
        # pure_physics = 完全不使用神经网络，仅用固定 Laplacian + CN
        # 预期误差最大，作为硬下界
        c.update({"use_physics_anchor": False, "use_mg_precond": False,
                  "use_learned_coarse": False, "use_dual_scale_gnn": False,
                  "use_virtual_nodes": False,  "dynamic_anchors": False,
                  "alpha_loc_init": 1.0, "alpha_coarse_init": 0.0})
    else:
        raise ValueError(f"未知 config: {name}")
    return c


def to_dev(batch, device):
    out = {}
    for k, v in batch.items():
        if   isinstance(v, torch.Tensor): out[k] = v.to(device)
        elif isinstance(v, dict):         out[k] = {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                                                    for kk, vv in v.items()}
        else:                             out[k] = v
    return out


# ── 单个 config 的 DDP 训练 + 验证 ────────────────────────────────────────────
def train_eval_ddp(name, base_cfg, data_path, args, rank, local_rank, world_size, device):
    # ── 数据 ──────────────────────────────────────────────────────────────────
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
    split = max(1, int(0.8 * len(keys)))
    tr_keys = keys[:split]
    va_keys  = keys[split:] or [keys[-1]]
    if args.max_train_samples:
        tr_keys = tr_keys[:args.max_train_samples]

    tr_ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                         trajectory_keys=tr_keys)
    va_ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                         trajectory_keys=va_keys)
    tr_sampler = DistributedSampler(tr_ds, world_size, rank) if world_size > 1 else None
    va_sampler = DistributedSampler(va_ds, world_size, rank) if world_size > 1 else None
    tr_loader  = create_dg_loader(tr_ds, batch_size=args.batch_size,
                                  shuffle=(tr_sampler is None), num_workers=2, sampler=tr_sampler)
    va_loader  = create_dg_loader(va_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2, sampler=va_sampler)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = PhysHGNet(cfg_for(name, base_cfg)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    alpha_p = [p for n, p in model.named_parameters() if "raw_alpha" in n or "raw_weights" in n]
    other_p = [p for n, p in model.named_parameters() if "raw_alpha" not in n and "raw_weights" not in n]
    opt = optim.Adam([{"params": other_p, "lr": args.lr},
                      {"params": alpha_p,  "lr": args.lr * 10}])

    # ── 训练 ──────────────────────────────────────────────────────────────────
    import time
    from tqdm import tqdm

    for ep in range(args.epochs):
        if tr_sampler: tr_sampler.set_epoch(ep)
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        bar = tqdm(tr_loader, desc=f"  [{name}] ep {ep+1:3d}/{args.epochs}",
                   disable=(rank != 0), leave=False)
        for batch in bar:
            batch = to_dev(batch, device)
            tgt   = batch["node_features"]
            opt.zero_grad()
            out  = model(batch)["u_final"]
            loss = (F.mse_loss(out[:, 1], tgt[:, 1]) +
                    F.mse_loss(out[:, -1], tgt[:, -1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        if rank == 0:
            n_bat = max(len(tr_loader), 1)
            print(f"  [{name}] Epoch {ep+1:3d}/{args.epochs} "
                  f"| train_loss={ep_loss/n_bat:.5f} "
                  f"| {time.time()-t0:.1f}s")

    # ── 验证（全部 rank 都跑，rank 0 汇总）──────────────────────────────────────
    model.eval()
    core = model.module if world_size > 1 else model
    tot_mse = tot_rne = tot_cg = n = 0.0
    with torch.no_grad():
        for batch in va_loader:
            batch = to_dev(batch, device)
            tgt   = batch["node_features"]
            out   = model(batch)["u_final"]
            tot_mse += mse_metric(out[:, -1], tgt[:, -1])
            tot_rne += rne_metric(out[:, -1], tgt[:, -1])
            tot_cg  += core.avg_cg_iters()
            n += 1
    n = max(n, 1)

    # 各卡指标做 all_reduce 求均值
    if world_size > 1:
        t = torch.tensor([tot_mse, tot_rne, tot_cg, n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tot_mse, tot_rne, tot_cg, n = t.tolist()

    row = None
    if rank == 0:
        core2 = model.module if world_size > 1 else model
        row = {
            "config":   name,
            "params":   core2.num_parameters(),
            "val_mse":  tot_mse / n,
            "val_rne":  tot_rne / n,
            "cg_iters": tot_cg  / n,
        }
        print(f"  [{name:14s}] MSE={row['val_mse']:.5f}  RNE={row['val_rne']:.4f}"
              f"  CG~{row['cg_iters']:.1f}  params={row['params']:,}")

    if world_size > 1:
        dist.barrier()
    return row


# ── 主函数 ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes",         type=int,   default=2000)
    p.add_argument("--data-dir",        type=str,   default="data_laser_hardening")
    p.add_argument("--data-path",       type=str,   default=None)
    p.add_argument("--epochs",          type=int,   default=15)
    p.add_argument("--batch-size",      type=int,   default=2)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--m-anchors",       type=int,   default=64)
    p.add_argument("--train-time-steps",type=int,   default=7)
    p.add_argument("--max-train-samples",type=int,  default=None)
    p.add_argument("--configs", nargs="+",
                   default=["full","no_c1","static_anchor","no_mg",
                            "no_c2","no_c3","no_vn","baseline"])
    p.add_argument("--out-dir", type=str, default="ablation_results")
    return p.parse_args()


def main():
    args   = parse_args()
    _ddp   = "RANK" in os.environ
    rank   = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if _ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    base_dir  = pathlib.Path(__file__).parent.resolve()
    data_path = (pathlib.Path(args.data_path) if args.data_path
                 else base_dir / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5")
    out_dir   = base_dir / args.out_dir
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 66)
        print(f"消融实验  GPUs={world_size}  epochs={args.epochs}  N={args.n_nodes}")
        print(f"configs: {args.configs}")
        print("=" * 66)

    base_cfg = {
        "spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
        "m_anchors": args.m_anchors, "residual_update_freq": 2,
    }

    rows = []
    for name in args.configs:
        if rank == 0:
            print(f"\n─── {name} ───")
        row = train_eval_ddp(name, base_cfg, str(data_path), args,
                             rank, local_rank, world_size, device)
        if rank == 0 and row:
            rows.append(row)

    # ── rank 0 打印汇总表 + 保存结果 ─────────────────────────────────────────
    if rank == 0 and rows:
        full_row = next((r for r in rows if r["config"] == "full"), rows[0])
        for r in rows:
            r["delta_rne"] = round(r["val_rne"] - full_row["val_rne"], 5)

        print("\n" + "=" * 74)
        print(f"{'config':14s}{'params':>10s}{'val_MSE':>11s}"
              f"{'val_RNE':>10s}{'CG_iters':>10s}{'Δ_RNE':>11s}")
        print("-" * 74)
        for r in rows:
            print(f"{r['config']:14s}{r['params']:>10,}{r['val_mse']:>11.5f}"
                  f"{r['val_rne']:>10.4f}{r['cg_iters']:>10.1f}{r['delta_rne']:>+11.5f}")
        print("=" * 74)
        print("Δ_RNE > 0 表示移除该组件会使误差变大（该组件有效）\n")

        # CSV
        keys_ = ["config","params","val_mse","val_rne","cg_iters","delta_rne"]
        with open(out_dir / "ablation_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys_)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys_})

        # 柱状图
        names = [r["config"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].bar(names, [r["val_rne"]  for r in rows], color=["#3377aa"]*len(names))
        ax[1].bar(names, [r["cg_iters"] for r in rows], color=["#22aa77"]*len(names))
        ax[0].set_ylabel("val RNE");    ax[0].set_title("各 config 精度")
        ax[1].set_ylabel("avg CG 迭代"); ax[1].set_title("各 config 求解迭代数")
        for a in ax:
            a.tick_params(axis="x", rotation=40)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_bar.png", dpi=150)
        print(f"结果已保存至 {out_dir}")

    if _ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
