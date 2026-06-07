"""
benchmark_scaling.py — 各规模 N 下的完整训练 + 评测（7 卡 DDP）

核心故事：
  对每个 N ∈ {4000, 6000, 8000, 10000}，用 DDP 训练两个模型：
    ✓ PhysHGNet：无矩阵 CG，O(N·nnz)，大 N 也能训练 + 推理
    ✗ DGNet    ：稠密 N×N LU，O(N²) 显存，大 N 训练/推理 OOM

  OOM 会被捕获并标记，PhysHGNet 继续运行——这是 TPAMI 的核心 scaling 优势。

每个 N 记录：
  - 训练：单 epoch 耗时(s)、训练峰值显存(MB)、OOM 标记
  - 评测：val MSE、val RNE、推理 ms/step、推理峰值显存(MB)

用法（推荐 7 卡 DDP）：
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \\
      benchmark_scaling.py \\
      --node-list 4000 6000 8000 10000 \\
      --data-dir data_laser_hardening \\
      --epochs 5 --batch-size 2 \\
      --out-dir bench_scaling

单卡降级（调试）：
  CUDA_VISIBLE_DEVICES=0 python3 benchmark_scaling.py \\
      --node-list 2000 4000 --epochs 3 --data-dir data_laser_hardening
"""
import os, csv, time, argparse, pathlib, traceback
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phys_hgnet import PhysHGNet
from dgnet import DGNet
from dataset import DGPdeDataset, create_dg_loader
from metrics import mse as mse_metric, rne as rne_metric


# ── DDP 工具 ──────────────────────────────────────────────────────────────────
def init_ddp():
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def barrier(world_size):
    if world_size > 1: dist.barrier()

def destroy_ddp(world_size):
    if world_size > 1: dist.destroy_process_group()

def to_dev(batch, device):
    out = {}
    for k, v in batch.items():
        if   isinstance(v, torch.Tensor): out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else: out[k] = v
    return out


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def make_loaders(data_path, train_steps, batch_size, rank, world_size):
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
    split    = max(1, int(0.8 * len(keys)))
    tr_keys  = keys[:split]
    va_keys  = keys[split:] or [keys[-1]]
    tr_ds = DGPdeDataset(data_path, train_time_steps=train_steps, trajectory_keys=tr_keys)
    va_ds = DGPdeDataset(data_path, train_time_steps=train_steps, trajectory_keys=va_keys)
    tr_samp = DistributedSampler(tr_ds, world_size, rank) if world_size > 1 else None
    va_samp = DistributedSampler(va_ds, world_size, rank) if world_size > 1 else None
    tr_loader = create_dg_loader(tr_ds, batch_size=batch_size, shuffle=(tr_samp is None),
                                 num_workers=2, sampler=tr_samp)
    va_loader = create_dg_loader(va_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2, sampler=va_samp)
    return tr_loader, va_loader, tr_samp


# ── 单 epoch 训练（返回 loss、耗时、峰值显存）────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, sampler, ep,
                    desc="train", rank=0):
    if sampler: sampler.set_epoch(ep)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from tqdm import tqdm
    tot_loss = 0.0; t0 = time.time()
    bar = tqdm(loader, desc=desc, disable=(rank != 0), leave=False)
    for batch in bar:
        batch = to_dev(batch, device)
        tgt   = batch["node_features"]
        optimizer.zero_grad()
        out   = model(batch)["u_final"]
        loss  = (F.mse_loss(out[:, 1], tgt[:, 1]) +
                 F.mse_loss(out[:, -1], tgt[:, -1]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    elapsed = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
               if device.type == "cuda" else float("nan"))
    return tot_loss / max(len(loader), 1), elapsed, peak_mb


# ── 验证 ──────────────────────────────────────────────────────────────────────
def validate(model, loader, device, world_size):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    tot_mse = tot_rne = n = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = to_dev(batch, device)
            tgt   = batch["node_features"]
            out   = model(batch)["u_final"]
            tot_mse += mse_metric(out[:, -1], tgt[:, -1])
            tot_rne += rne_metric(out[:, -1], tgt[:, -1])
            n += 1
    if world_size > 1:
        t = torch.tensor([tot_mse, tot_rne, n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tot_mse, tot_rne, n = t.tolist()
    n = max(n, 1)
    return tot_mse / n, tot_rne / n


# ── 推理计时（rank 0 单卡，排除通信干扰）─────────────────────────────────────
def inference_timing(model_bare, data_path, device, t_bench=11, repeats=3):
    """加载一条轨迹，在 rank-0 的本地 GPU 上计时推理。"""
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
        g = f[keys[-1]]   # 用最后一条（独立测试轨迹）
        nodes    = torch.tensor(g["nodes"][:],         dtype=torch.float32)
        edges    = torch.tensor(g["edges"][:],         dtype=torch.long)
        faces    = torch.tensor(g["faces"][:],         dtype=torch.long)
        feats    = torch.tensor(g["node_features"][:], dtype=torch.float32)
        srcs     = torch.tensor(g["source_terms"][:],  dtype=torch.float32)
        tpts     = torch.tensor(g["time_points"][:],   dtype=torch.float32)
        node_type = (torch.tensor(g["node_type"][:], dtype=torch.long)
                     if "node_type" in g else torch.zeros(nodes.shape[0], dtype=torch.long))
        node_vol  = (torch.tensor(g["node_volumes"][:], dtype=torch.float32)
                     if "node_volumes" in g else torch.ones(nodes.shape[0]))
    T = min(t_bench, feats.shape[0])
    batch = {
        "nodes": nodes.to(device), "edges": edges.to(device),
        "faces": faces.to(device), "node_volumes": node_vol.to(device),
        "node_type": node_type.to(device),
        "source_terms": srcs[:T].unsqueeze(0).to(device),
        "initial_conditions": feats[0:1].to(device),
        "node_features": feats[:T].unsqueeze(0).to(device),
        "time_points": tpts[:T].to(device), "boundary_info": {},
    }
    model_bare.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    with torch.no_grad():
        model_bare(batch)   # warmup
        if device.type == "cuda": torch.cuda.synchronize()
        for _ in range(repeats):
            if device.type == "cuda":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); model_bare(batch); e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter(); model_bare(batch)
                times.append((time.perf_counter() - t0) * 1e3)
    infer_ms   = float(np.median(times))
    infer_peak = (torch.cuda.max_memory_allocated(device) / 1024**2
                  if device.type == "cuda" else float("nan"))
    steps = max(T - 1, 1)
    return round(infer_ms, 2), round(infer_ms / steps, 3), round(infer_peak, 1)


# ── 单个 N 的完整训练 + 评测 ──────────────────────────────────────────────────
def benchmark_one_N(N_req, args, base, rank, local_rank, world_size, device):
    data_path = base / args.data_dir / f"pde_trajectories_{N_req}.h5"
    if not data_path.exists():
        if rank == 0:
            print(f"\n  [N={N_req}] ⚠ 数据不存在: {data_path}，跳过")
        return []

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  N = {N_req}  （{world_size} 卡 DDP 训练）")
        print(f"{'='*60}")

    tr_loader, va_loader, tr_samp = make_loaders(
        str(data_path), args.train_steps, args.batch_size, rank, world_size)

    rows = []

    # ── 依次测试两个模型 ──────────────────────────────────────────────────────
    model_specs = [
        ("PhysHGNet",
         PhysHGNet({"spatial_dim":2,"feature_dim":1,"output_dim":1,
                    "m_anchors":args.m_anchors,"use_mg_precond":True,
                    "use_physics_anchor":True})),
        ("DGNet",
         DGNet({"spatial_dim":2,"feature_dim":1,"output_dim":1,
                "residual_hidden_dim":128,"residual_num_layers":5,
                "operator_hidden_dim":64,"operator_num_layers":3})),
    ]

    for mdl_name, model_raw in model_specs:
        row = {"model": mdl_name, "N": N_req,
               "params": model_raw.num_parameters(), "note": ""}
        if rank == 0:
            print(f"\n  ── {mdl_name} ({row['params']:,} params) ──")

        try:
            # ── DDP 训练 ──────────────────────────────────────────────────────
            model = model_raw.to(device)
            if world_size > 1:
                model = DDP(model, device_ids=[local_rank],
                            find_unused_parameters=True)

            alpha_p = [p for n, p in model.named_parameters()
                       if "raw_alpha" in n or "raw_weights" in n]
            other_p = [p for n, p in model.named_parameters()
                       if "raw_alpha" not in n and "raw_weights" not in n]
            opt = optim.Adam([{"params": other_p, "lr": args.lr},
                              {"params": alpha_p,  "lr": args.lr * 10}])

            tr_times = []; tr_peaks = []
            for ep in range(args.epochs):
                ep_desc = (f"  [{mdl_name} N={N_req}] "
                           f"ep{ep+1}/{args.epochs}")
                ep_loss, ep_time, ep_peak = train_one_epoch(
                    model, tr_loader, opt, device, tr_samp, ep,
                    desc=ep_desc, rank=rank)
                tr_times.append(ep_time); tr_peaks.append(ep_peak)
                if rank == 0:
                    print(f"    Epoch {ep+1:2d}/{args.epochs} "
                          f"| loss={ep_loss:.5f} "
                          f"| {ep_time:.1f}s "
                          f"| peak={ep_peak:.0f}MB")
            barrier(world_size)

            # ── 验证 ──────────────────────────────────────────────────────────
            val_mse, val_rne = validate(model, va_loader, device, world_size)

            # ── 推理计时（rank 0 本地单卡）────────────────────────────────────
            infer_ms = infer_mps = infer_peak = float("nan")
            if rank == 0:
                core = model.module if world_size > 1 else model
                infer_ms, infer_mps, infer_peak = inference_timing(
                    core, str(data_path), device, args.t_bench, args.infer_repeats)

            row.update({
                "train_s_per_epoch": round(np.mean(tr_times), 1),
                "train_peak_mb":     round(np.max(tr_peaks), 1),
                "val_mse":           round(val_mse, 6),
                "val_rne":           round(val_rne, 5),
                "infer_ms":          infer_ms,
                "infer_ms_per_step": infer_mps,
                "infer_peak_mb":     infer_peak,
            })
            if rank == 0:
                print(f"    ✓ val_MSE={val_mse:.5f}  val_RNE={val_rne:.4f}"
                      f"  infer={infer_ms:.1f}ms  infer_peak={infer_peak:.0f}MB")

        except RuntimeError as ex:
            is_oom = "out of memory" in str(ex).lower()
            row["note"] = "OOM" if is_oom else str(ex)[:80]
            if rank == 0:
                if is_oom:
                    print(f"    ✗ OOM！N×N 稠密矩阵超出显存——"
                          f"这正是 PhysHGNet 无矩阵 CG 的优势所在")
                else:
                    print(f"    ✗ ERROR: {row['note']}")
                    traceback.print_exc()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        finally:
            del model, model_raw
            if device.type == "cuda":
                torch.cuda.empty_cache()
            barrier(world_size)

        rows.append(row)

    return rows


# ── 绘图 ──────────────────────────────────────────────────────────────────────
def make_plots(rows, out_dir):
    rows = sorted(rows, key=lambda r: r["N"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    metrics = [
        ("train_s_per_epoch", "训练耗时 s/epoch",  "s"),
        ("train_peak_mb",     "训练峰值显存",       "MB"),
        ("val_rne",           "验证集 RNE",         "RNE"),
        ("infer_ms_per_step", "推理延迟 ms/step",   "ms"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        for mdl, col, mk in [("PhysHGNet","#2a9d8f","o"), ("DGNet","#e76f51","s")]:
            sub = [r for r in rows if r["model"] == mdl]
            Ns  = [r["N"] for r in sub]
            ys  = [r.get(key, float("nan")) for r in sub]
            ax.plot(Ns, ys, f"{mk}-", label=mdl, color=col, linewidth=2, markersize=7)
            for r in sub:
                if r.get("note") == "OOM":
                    ax.annotate("OOM", (r["N"], 0), color="red",
                                fontsize=9, ha="center", va="bottom",
                                arrowprops=dict(arrowstyle="->", color="red"),
                                xytext=(r["N"], max(ys[i] for i,v in enumerate(ys)
                                                   if not np.isnan(v)) * 0.5
                                        if any(not np.isnan(v) for v in ys) else 1))
        ax.set_title(title); ax.set_xlabel("节点数 N"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("PhysHGNet vs DGNet — Scaling with N", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_nodes.png", dpi=150)
    plt.close(fig)
    print(f"  图表已保存: {out_dir / 'scaling_nodes.png'}")


def print_table(rows):
    rows = sorted(rows, key=lambda r: (r["N"], r["model"]))
    cols = ["model","N","params","train_s_per_epoch","train_peak_mb",
            "val_mse","val_rne","infer_ms_per_step","infer_peak_mb","note"]
    header = (f"{'model':10s}{'N':>7s}{'params':>10s}"
              f"{'tr_s/ep':>9s}{'tr_MB':>8s}"
              f"{'val_MSE':>10s}{'val_RNE':>9s}"
              f"{'inf_ms/s':>10s}{'inf_MB':>8s}{'note':>6s}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(k, w, d=1):
            v = r.get(k, "—")
            if isinstance(v, float) and np.isnan(v): return f"{'—':>{w}}"
            if isinstance(v, float): return f"{v:>{w}.{d}f}"
            return f"{str(v):>{w}}"
        print(f"{r['model']:10s}{r['N']:>7d}{r['params']:>10,}"
              f"{fmt('train_s_per_epoch',9,1)}"
              f"{fmt('train_peak_mb',8,0)}"
              f"{fmt('val_mse',10,5)}"
              f"{fmt('val_rne',9,4)}"
              f"{fmt('infer_ms_per_step',10,2)}"
              f"{fmt('infer_peak_mb',8,0)}"
              f"{r.get('note',''):>6s}")
    print("=" * len(header))


# ── 主函数 ────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-list",   type=int, nargs="+", default=[4000,6000,8000,10000])
    ap.add_argument("--data-dir",    type=str, default="data_laser_hardening")
    ap.add_argument("--epochs",      type=int, default=5,
                    help="每个 N 训练的 epoch 数（scaling 实验不需要完整收敛）")
    ap.add_argument("--batch-size",  type=int, default=2)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--m-anchors",   type=int, default=64)
    ap.add_argument("--train-steps", type=int, default=7)
    ap.add_argument("--t-bench",     type=int, default=11,
                    help="推理计时使用的时间步数")
    ap.add_argument("--infer-repeats",type=int, default=3)
    ap.add_argument("--out-dir",     type=str, default="bench_scaling")
    return ap.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = init_ddp()
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    base    = pathlib.Path(__file__).parent.resolve()
    out_dir = base / args.out_dir

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print(f"  Scaling Benchmark  GPUs={world_size}  epochs/N={args.epochs}")
        print(f"  N_list = {args.node_list}")
        print(f"  策略: 对每个 N 用 {world_size} 卡 DDP 完整训练 + 评测")
        print("=" * 60)

    all_rows = []
    for N_req in args.node_list:
        rows = benchmark_one_N(N_req, args, base, rank, local_rank, world_size, device)
        all_rows.extend(rows)
        barrier(world_size)

    if rank == 0 and all_rows:
        print_table(all_rows)
        make_plots(all_rows, out_dir)
        keys_ = ["model","N","params","train_s_per_epoch","train_peak_mb",
                 "val_mse","val_rne","infer_ms_per_step","infer_peak_mb","note"]
        with open(out_dir / "scaling_nodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys_, extrasaction="ignore")
            w.writeheader(); w.writerows(all_rows)
        print(f"\n  CSV 已保存: {out_dir / 'scaling_nodes.csv'}")

    destroy_ddp(world_size)


if __name__ == "__main__":
    main()
