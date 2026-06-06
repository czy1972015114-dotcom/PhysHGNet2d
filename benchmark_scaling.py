"""
benchmark_scaling.py — Scaling comparison of PhysHGNet vs DGNet.

Two studies (select with --mode):

  --mode anchor   Fix the mesh (one .h5) and sweep PhysHGNet's anchor count m.
                  Shows how inference time / peak memory / accuracy trade off
                  with the number of anchors. DGNet (no anchors) is reported as
                  a flat reference line on the same axes.

  --mode nodes    Sweep the mesh size N over several .h5 files and run BOTH
                  PhysHGNet and DGNet. This is the headline comparison: DGNet
                  factorises a dense N×N matrix (O(N^2) memory, O(N^3) factor),
                  while PhysHGNet uses matrix-free CG on a fine+coarse hierarchy
                  (≈O(N) memory). DGNet typically slows down / OOMs first.

For every (model, setting) we record: #parameters, peak GPU memory (MB),
median rollout time (s) and per-step time (ms), and relative L2 error at the
final step. Error is only meaningful if you pass trained checkpoints; otherwise
weights are random and ONLY time/memory are interpretable (the script prints a
clear warning in that case).

Outputs (under --out-dir): scaling_<mode>.csv + PNG plots.

Examples
--------
  # anchor-count scaling on the 2000-node mesh
  python benchmark_scaling.py --mode anchor --n-nodes 2000 \
      --anchors 16 32 64 128 256 \
      --physhgnet-ckpt checkpoints/phys_hgnet/best_2000.pth

  # node scaling, both models (generate these meshes first, see run_all.sh)
  python benchmark_scaling.py --mode nodes \
      --node-list 500 1000 2000 4000 8000 \
      --data-dir data_laser_hardening
"""
import os
import gc as _gc
import csv
import time
import argparse
import pathlib
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phys_hgnet import PhysHGNet, DEFAULT_CONFIG
from dgnet import DGNet


# ──────────────────────────────────────────────────────────────
def load_trajectory(data_path, traj_index, device, t_bench, batch_size):
    import h5py
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
        g = f[keys[traj_index]]
        nodes = torch.from_numpy(g["nodes"][:]).float()
        edges = torch.from_numpy(g["edges"][:]).long()
        faces = torch.from_numpy(g["faces"][:]).long()
        node_features = torch.from_numpy(g["node_features"][:]).float()
        source_terms = torch.from_numpy(g["source_terms"][:]).float()
        time_points = torch.from_numpy(g["time_points"][:]).float()
        node_type = (torch.from_numpy(g["node_type"][:]).long()
                     if "node_type" in g else torch.zeros(nodes.shape[0], dtype=torch.long))
        node_volumes = (torch.from_numpy(g["node_volumes"][:]).float()
                        if "node_volumes" in g else None)

    T = min(t_bench, node_features.shape[0])
    nf = node_features[:T]
    st = source_terms[:T]
    tp = time_points[:T]
    B = batch_size
    batch = {
        "nodes": nodes.to(device),
        "edges": edges.to(device),
        "faces": faces.to(device),
        "node_type": node_type.to(device),
        "node_volumes": (node_volumes.to(device)
                         if node_volumes is not None
                         else torch.ones(nodes.shape[0], device=device)),
        "boundary_info": {},
        "source_terms": st.unsqueeze(0).repeat(B, 1, 1, 1).to(device),
        "time_points": tp.to(device),
        "initial_conditions": nf[0].unsqueeze(0).repeat(B, 1, 1).to(device),
        "node_features": nf.unsqueeze(0).repeat(B, 1, 1, 1).to(device),
    }
    return batch, nodes.shape[0], T


def rel_err_final(out, batch):
    pred = out["u_final"][:, -1]
    tgt = batch["node_features"][:, -1]
    return (torch.norm(pred - tgt) / torch.norm(tgt).clamp(min=1e-8)).item()


def time_and_mem(model, batch, device, repeats=3):
    is_cuda = device.type == "cuda"
    # warmup
    with torch.no_grad():
        _ = model(batch)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    err = None
    with torch.no_grad():
        for _ in range(repeats):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(batch)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            err = rel_err_final(out, batch)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2) if is_cuda else float("nan")
    return float(np.median(times)), peak_mb, err


def load_state(model, ckpt_path, device):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("model_state", ck.get("model_state_dict", ck))
    model.load_state_dict(state, strict=False)
    return True


def build_physhgnet(N, m_anchors, device, ckpt, uncap=False):
    cfg = dict(DEFAULT_CONFIG)
    cfg["m_anchors"] = m_anchors
    if uncap:                       # let m actually reach the requested value
        cfg["anchor_cap_ratio"] = 1
        cfg["anchor_cap_max"] = 10**9
    model = PhysHGNet(cfg).to(device)
    loaded = load_state(model, ckpt, device)
    model.eval()
    return model, loaded


def build_dgnet(device, ckpt):
    cfg = {"spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
           "operator_type": "laplace", "operator_hidden_dim": 64,
           "operator_num_layers": 3, "residual_hidden_dim": 128,
           "residual_num_layers": 5}
    model = DGNet(cfg).to(device)
    loaded = load_state(model, ckpt, device)
    model.eval()
    return model, loaded


def free(model):
    del model
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────
def run_anchor_mode(args, device, out_dir):
    base = pathlib.Path(__file__).parent.resolve()
    data_path = (pathlib.Path(args.data_path) if args.data_path
                 else base / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5")
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    batch, N, T = load_trajectory(data_path, args.traj_index, device,
                                  args.t_bench, args.batch_size)
    print(f"[anchor mode] N={N}, T={T}, B={args.batch_size}, device={device}")

    rows = []
    # DGNet reference (no anchors)
    dg, dg_loaded = build_dgnet(device, args.dgnet_ckpt)
    t_dg, mem_dg, err_dg = time_and_mem(dg, batch, device, args.repeats)
    p_dg = dg.num_parameters() if hasattr(dg, "num_parameters") else sum(p.numel() for p in dg.parameters())
    free(dg)
    print(f"  DGNet           : {t_dg*1000:8.1f} ms  mem={mem_dg:8.1f} MB  err={err_dg:.4f}")

    for m in args.anchors:
        try:
            ph, ph_loaded = build_physhgnet(N, m, device, args.physhgnet_ckpt, uncap=True)
            t, mem, err = time_and_mem(ph, batch, device, args.repeats)
            params = ph.num_parameters()
            m_eff = ph._m_target(N)
            free(ph)
            rows.append({"m_anchors": m, "m_effective": m_eff, "params": params,
                         "time_s": t, "per_step_ms": 1000 * t / max(1, T - 1),
                         "peak_mem_mb": mem, "rel_err": err})
            print(f"  PhysHGNet m={m:<4d}(eff {m_eff:<4d}): {t*1000:8.1f} ms  "
                  f"mem={mem:8.1f} MB  err={err:.4f}")
        except RuntimeError as e:
            print(f"  PhysHGNet m={m}: FAILED ({e})")

    # CSV
    csv_path = out_dir / "scaling_anchor.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {csv_path}")

    ms = [r["m_effective"] for r in rows]
    note = "" if (dg_loaded or args.physhgnet_ckpt) else "  (RANDOM weights: error not meaningful)"
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    axs[0].plot(ms, [r["per_step_ms"] for r in rows], "o-", label="PhysHGNet")
    axs[0].axhline(1000 * t_dg / max(1, T - 1), color="r", ls="--", label="DGNet")
    axs[0].set_xlabel("# anchors (effective)"); axs[0].set_ylabel("per-step time (ms)")
    axs[0].set_title("Inference time vs anchors"); axs[0].legend()
    axs[1].plot(ms, [r["peak_mem_mb"] for r in rows], "o-", label="PhysHGNet")
    axs[1].axhline(mem_dg, color="r", ls="--", label="DGNet")
    axs[1].set_xlabel("# anchors (effective)"); axs[1].set_ylabel("peak GPU mem (MB)")
    axs[1].set_title("Memory vs anchors"); axs[1].legend()
    axs[2].plot(ms, [r["rel_err"] for r in rows], "o-", label="PhysHGNet")
    axs[2].axhline(err_dg, color="r", ls="--", label="DGNet")
    axs[2].set_xlabel("# anchors (effective)"); axs[2].set_ylabel("rel L2 error (final)")
    axs[2].set_title("Accuracy vs anchors" + note); axs[2].legend()
    fig.suptitle(f"PhysHGNet anchor scaling on N={N} mesh", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "scaling_anchor.png", dpi=130)
    plt.close(fig)
    print(f"[plot] {out_dir/'scaling_anchor.png'}")


def run_nodes_mode(args, device, out_dir):
    base = pathlib.Path(__file__).parent.resolve()
    rows = []
    for N_req in args.node_list:
        data_path = base / args.data_dir / f"pde_trajectories_{N_req}.h5"
        if not data_path.exists():
            print(f"  [skip] missing {data_path}")
            continue
        batch, N, T = load_trajectory(data_path, args.traj_index, device,
                                      args.t_bench, args.batch_size)
        m = min(args.max_anchors, max(16, int(N // 8)))
        print(f"[N={N}] T={T} B={args.batch_size}  (PhysHGNet m≈{m})")

        # PhysHGNet
        try:
            ph, ph_loaded = build_physhgnet(N, m, device,
                                            args.physhgnet_ckpt, uncap=True)
            t, mem, err = time_and_mem(ph, batch, device, args.repeats)
            free(ph)
            rows.append({"model": "PhysHGNet", "N": N, "m": m, "time_s": t,
                         "per_step_ms": 1000 * t / max(1, T - 1),
                         "peak_mem_mb": mem, "rel_err": err})
            print(f"    PhysHGNet : {t*1000:8.1f} ms  mem={mem:8.1f} MB  err={err:.4f}")
        except RuntimeError as e:
            print(f"    PhysHGNet : FAILED ({e})")
            rows.append({"model": "PhysHGNet", "N": N, "m": m, "time_s": float("nan"),
                         "per_step_ms": float("nan"), "peak_mem_mb": float("nan"),
                         "rel_err": float("nan")})

        # DGNet
        try:
            dg, dg_loaded = build_dgnet(device, args.dgnet_ckpt)
            t, mem, err = time_and_mem(dg, batch, device, args.repeats)
            free(dg)
            rows.append({"model": "DGNet", "N": N, "m": 0, "time_s": t,
                         "per_step_ms": 1000 * t / max(1, T - 1),
                         "peak_mem_mb": mem, "rel_err": err})
            print(f"    DGNet     : {t*1000:8.1f} ms  mem={mem:8.1f} MB  err={err:.4f}")
        except RuntimeError as e:
            print(f"    DGNet     : FAILED ({e})  <-- likely dense-LU OOM")
            rows.append({"model": "DGNet", "N": N, "m": 0, "time_s": float("nan"),
                         "per_step_ms": float("nan"), "peak_mem_mb": float("nan"),
                         "rel_err": float("nan")})

    if not rows:
        print("No data files found. Generate meshes first (see run_all.sh).")
        return

    csv_path = out_dir / "scaling_nodes.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[csv] {csv_path}")

    def series(model, key):
        xs = sorted(set(r["N"] for r in rows if r["model"] == model))
        ys = []
        for x in xs:
            v = [r[key] for r in rows if r["model"] == model and r["N"] == x]
            ys.append(v[0] if v else float("nan"))
        return xs, ys

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for model, mk in [("PhysHGNet", "o-"), ("DGNet", "s--")]:
        x, y = series(model, "per_step_ms"); axs[0].plot(x, y, mk, label=model)
        x, y = series(model, "peak_mem_mb"); axs[1].plot(x, y, mk, label=model)
        x, y = series(model, "rel_err"); axs[2].plot(x, y, mk, label=model)
    for a, ttl, yl in [(axs[0], "Per-step time", "ms"),
                       (axs[1], "Peak GPU memory", "MB"),
                       (axs[2], "Final rel L2 error", "")]:
        a.set_xlabel("mesh size N"); a.set_ylabel(yl); a.set_title(ttl)
        a.set_xscale("log"); a.legend()
    axs[0].set_yscale("log"); axs[1].set_yscale("log")
    fig.suptitle("PhysHGNet vs DGNet — scaling with mesh size", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "scaling_nodes.png", dpi=130)
    plt.close(fig)
    print(f"[plot] {out_dir/'scaling_nodes.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["anchor", "nodes"], required=True)
    ap.add_argument("--data-dir", type=str, default="data_laser_hardening")
    ap.add_argument("--data-path", type=str, default=None)
    ap.add_argument("--n-nodes", type=int, default=2000, help="(anchor mode) mesh to use")
    ap.add_argument("--anchors", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256], help="(anchor mode) sweep")
    ap.add_argument("--node-list", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000], help="(nodes mode) meshes")
    ap.add_argument("--max-anchors", type=int, default=256, help="(nodes mode) cap on m")
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--t-bench", type=int, default=11, help="timesteps used for timing")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--physhgnet-ckpt", type=str, default=None)
    ap.add_argument("--dgnet-ckpt", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="bench_scaling")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = pathlib.Path(__file__).parent.resolve()
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (args.physhgnet_ckpt or args.dgnet_ckpt):
        print("[warn] no checkpoints given -> random weights; ONLY time/memory are "
              "meaningful, rel_err is not.")

    if args.mode == "anchor":
        run_anchor_mode(args, device, out_dir)
    else:
        run_nodes_mode(args, device, out_dir)


if __name__ == "__main__":
    main()
