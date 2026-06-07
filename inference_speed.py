"""
inference_speed.py — real-time inference-speed benchmark (Task 5.1).

DGNet's selling point is fast inference once its operator is factorised; we
measure the same thing for PhysHGNet on INDEPENDENT trajectories (ones not used
to fit the timing, loaded straight from the HDF5 file) and compare.

For each model we report, over `--repeats` timed roll-outs (after warm-up):
  * total roll-out latency (median, p10, p90)   [ms]
  * per-step latency                            [ms/step]
  * throughput                                  [steps/s] and [node-steps/s]
  * (PhysHGNet) average CG iterations per solve
Timing uses CUDA events with torch.cuda.synchronize so it is wall-clock honest.

Usage:
  CUDA_VISIBLE_DEVICES=0 python inference_speed.py \
      --n-nodes 2000 --data-dir data_laser_hardening --traj-index -1 \
      --repeats 10 --warmup 3 --out-dir speed_results \
      --physhgnet-ckpt checkpoints/phys_hgnet/best_2000.pth \
      --dgnet-ckpt     checkpoints/dgnet/best_2000.pth
"""
import os
import csv
import time
import argparse
import pathlib

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phys_hgnet import PhysHGNet
from dgnet import DGNet


def load_trajectory(data_path, traj_index, device, max_steps=None):
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
        key = keys[traj_index]
        g = f[key]
        nodes = torch.tensor(g["nodes"][:], dtype=torch.float32)
        edges = torch.tensor(g["edges"][:], dtype=torch.long)
        faces = torch.tensor(g["faces"][:], dtype=torch.long)
        feats = torch.tensor(g["node_features"][:], dtype=torch.float32)   # [T,N,1]
        srcs = torch.tensor(g["source_terms"][:], dtype=torch.float32)
        tpts = torch.tensor(g["time_points"][:], dtype=torch.float32)
        node_type = (torch.tensor(g["node_type"][:], dtype=torch.long)
                     if "node_type" in g else torch.zeros(nodes.shape[0], dtype=torch.long))
        node_vol = (torch.tensor(g["node_volumes"][:], dtype=torch.float32)
                    if "node_volumes" in g else torch.ones(nodes.shape[0]))
    T = feats.shape[0] if max_steps is None else min(max_steps, feats.shape[0])
    batch = {
        "nodes": nodes.to(device), "edges": edges.to(device), "faces": faces.to(device),
        "node_volumes": node_vol.to(device), "node_type": node_type.to(device),
        "source_terms": srcs[:T].unsqueeze(0).to(device),
        "initial_conditions": feats[0:1].to(device),
        "node_features": feats[:T].unsqueeze(0).to(device),
        "time_points": tpts[:T].to(device),
        "boundary_info": {},
    }
    return batch, key, T, nodes.shape[0]


def time_model(model, batch, repeats, warmup, device):
    model.eval()
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                model(batch)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))            # ms
            else:
                t0 = time.perf_counter()
                model(batch)
                times.append((time.perf_counter() - t0) * 1e3)
    return np.array(times)


def summarise(times, T, N):
    steps = max(T - 1, 1)
    med = float(np.median(times))
    return {
        "total_ms_median": med,
        "total_ms_p10": float(np.percentile(times, 10)),
        "total_ms_p90": float(np.percentile(times, 90)),
        "ms_per_step": med / steps,
        "steps_per_s": 1e3 * steps / med,
        "node_steps_per_s": 1e3 * steps * N / med,
    }


def maybe_load(model, ckpt):
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model_state", sd), strict=False)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=2000)
    ap.add_argument("--data-dir", type=str, default="data_laser_hardening")
    ap.add_argument("--data-path", type=str, default=None)
    ap.add_argument("--traj-index", type=int, default=-1, help="independent trajectory index")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--m-anchors", type=int, default=64)
    ap.add_argument("--physhgnet-ckpt", type=str, default=None)
    ap.add_argument("--dgnet-ckpt", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="speed_results")
    args = ap.parse_args()

    base = pathlib.Path(__file__).parent.resolve()
    data_path = (pathlib.Path(args.data_path) if args.data_path
                 else base / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5")
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch, key, T, N = load_trajectory(data_path, args.traj_index, device, args.max_steps)
    print(f"Independent trajectory '{key}': N={N}, T={T} on {device}")

    ph = PhysHGNet({"m_anchors": args.m_anchors, "use_physics_anchor": True,
                    "dynamic_anchors": True, "use_mg_precond": True}).to(device)
    dg = DGNet({"spatial_dim": 2, "feature_dim": 1, "output_dim": 1}).to(device)
    print(f"PhysHGNet ckpt loaded: {maybe_load(ph, args.physhgnet_ckpt)} | "
          f"DGNet ckpt loaded: {maybe_load(dg, args.dgnet_ckpt)}")

    rows = []
    for name, model in [("PhysHGNet", ph), ("DGNet", dg)]:
        try:
            times = time_model(model, batch, args.repeats, args.warmup, device)
            row = {"model": name, "N": N, "T": T, **summarise(times, T, N)}
            if name == "PhysHGNet":
                row["avg_cg_iters"] = round(model.avg_cg_iters(), 2)
            print(f"  {name:10s} {row['total_ms_median']:.1f} ms/rollout  "
                  f"{row['ms_per_step']:.2f} ms/step  {row['steps_per_s']:.1f} steps/s")
        except RuntimeError as ex:
            row = {"model": name, "N": N, "T": T, "total_ms_median": float("nan"),
                   "note": "OOM" if "out of memory" in str(ex).lower() else str(ex)[:60]}
            print(f"  {name:10s} FAILED: {row['note']}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)

    csv_path = out_dir / "inference_speed.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # bar charts: per-step latency and throughput
    names = [r["model"] for r in rows]
    msps = [r.get("ms_per_step", float("nan")) for r in rows]
    thr = [r.get("steps_per_s", float("nan")) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(names, msps, color=["#2a7", "#37a"])
    ax[0].set_ylabel("ms / step"); ax[0].set_title(f"Per-step latency (N={N})")
    ax[1].bar(names, thr, color=["#2a7", "#37a"])
    ax[1].set_ylabel("steps / s"); ax[1].set_title("Throughput (higher=better)")
    fig.tight_layout()
    fig.savefig(out_dir / "inference_speed.png", dpi=150)
    print(f"Saved {csv_path} and inference_speed.png")


if __name__ == "__main__":
    main()
