"""
ablation_experiment.py — Single-GPU ablation study for PhysHGNet.

Trains the model under several component configurations on the SAME data split,
then reports validation relative error, #parameters, and the change (Δ) relative
to the full model. This isolates the contribution of each innovation:

  full            all components ON (dynamic physics anchors + C2 + C3 + VN)
  no_c1           C1 OFF  -> plain geometric FPS anchors
  static_anchor   C1 ON but anchors frozen at t=0 (isolates the dynamic update)
  no_c2           C2 OFF  -> fixed (1/dist) coarse operator instead of learned
  no_c3           C3 OFF  -> physics path only, no dual-scale GNN correction
  no_vn           virtual nodes OFF (C3 sub-ablation)
  baseline        C1+C2+C3 all OFF (≈ Structured DGNet)

Everything runs in a single process (no DDP) for reproducibility and simple
orchestration. Use a small --epochs for a quick study; increase for final numbers.

Usage
-----
  python ablation_experiment.py --n-nodes 2000 --epochs 8 --batch-size 2
  python ablation_experiment.py --n-nodes 2000 --epochs 8 \
      --configs full no_c1 static_anchor no_c2 no_c3 no_vn baseline
"""
import os
import csv
import time
import argparse
import pathlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phys_hgnet import PhysHGNet, DEFAULT_CONFIG
from dataset import DGPdeDataset, create_dg_loader


# Each entry overrides DEFAULT_CONFIG.
ABLATION_CONFIGS = {
    "full":          dict(use_physics_anchor=True,  dynamic_anchors=True,
                          use_learned_coarse=True,  use_dual_scale_gnn=True,
                          use_virtual_nodes=True),
    "no_c1":         dict(use_physics_anchor=False, dynamic_anchors=True),
    "static_anchor": dict(use_physics_anchor=True,  dynamic_anchors=False),
    "no_c2":         dict(use_learned_coarse=False),
    "no_c3":         dict(use_dual_scale_gnn=False),
    "no_vn":         dict(use_virtual_nodes=False),
    "baseline":      dict(use_physics_anchor=False, use_learned_coarse=False,
                          use_dual_scale_gnn=False, use_virtual_nodes=False),
}


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


def rel_err(pred, target):
    with torch.no_grad():
        return (torch.norm(pred - target) / torch.norm(target).clamp(min=1e-8)).item()


def loss_fn(out, target):
    l1 = F.mse_loss(out["u_final"][:, 1], target[:, 1])
    lT = F.mse_loss(out["u_final"][:, -1], target[:, -1])
    return l1 + lT


def make_config(name, args):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "spatial_dim": 2, "feature_dim": 1, "output_dim": 1,
        "operator_type": "laplace",
        "m_anchors": args.m_anchors, "coarse_num_layers": args.coarse_layers,
        "k_virtual_nodes": args.k_vn,
        "residual_update_freq": args.update_freq,
    })
    cfg.update(ABLATION_CONFIGS[name])
    return cfg


def train_eval_one(name, args, device, train_loader, val_loader):
    cfg = make_config(name, args)
    model = PhysHGNet(cfg).to(device)
    n_params = model.num_parameters()
    print(f"\n=== [{name}] ===")
    print(model.ablation_summary())

    alpha_params = [p for n, p in model.named_parameters() if "raw_alpha" in n]
    other = [p for n, p in model.named_parameters() if "raw_alpha" not in n]
    opt = optim.Adam([{"params": other, "lr": args.lr},
                      {"params": alpha_params, "lr": args.lr * 10}])
    sched = optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.3)

    best_val = float("inf")
    best_err = float("nan")
    history = defaultdict(list)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tl = []
        for batch in train_loader:
            batch = to_device(batch, device)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch["node_features"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
        sched.step()

        model.eval()
        vl, ve = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                out = model(batch)
                vl.append(loss_fn(out, batch["node_features"]).item())
                ve.append(rel_err(out["u_final"][:, -1], batch["node_features"][:, -1]))
        v_loss, v_err = float(np.mean(vl)), float(np.mean(ve))
        history["train_loss"].append(float(np.mean(tl)))
        history["val_loss"].append(v_loss)
        history["val_err"].append(v_err)
        if v_loss < best_val:
            best_val, best_err = v_loss, v_err
        print(f"  epoch {ep+1:2d}/{args.epochs} | train={np.mean(tl):.5f} "
              f"val={v_loss:.5f} relErr={v_err:.4f} | {time.time()-t0:.1f}s")

    return {"config": name, "params": n_params,
            "best_val_loss": best_val, "best_val_rel_err": best_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=2000)
    ap.add_argument("--data-dir", type=str, default="data_laser_hardening")
    ap.add_argument("--data-path", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--m-anchors", type=int, default=64)
    ap.add_argument("--coarse-layers", type=int, default=4)
    ap.add_argument("--k-vn", type=int, default=4)
    ap.add_argument("--update-freq", type=int, default=2)
    ap.add_argument("--train-time-steps", type=int, default=7)
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="cap #training chunks for a faster study")
    ap.add_argument("--configs", type=str, nargs="+",
                    default=list(ABLATION_CONFIGS.keys()))
    ap.add_argument("--out-dir", type=str, default="ablation_results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = pathlib.Path(__file__).parent.resolve()
    data_path = (pathlib.Path(args.data_path) if args.data_path
                 else base / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path}\nGenerate it: python generate_laser_data_aligned.py "
            f"--n_nodes {args.n_nodes} --out_dir {args.data_dir}")
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
    split = max(1, int(0.8 * len(keys)))
    train_keys, val_keys = keys[:split], (keys[split:] or [keys[-1]])
    print(f"[data] {data_path} | train traj={len(train_keys)} val traj={len(val_keys)} "
          f"| device={device}")

    train_ds = DGPdeDataset(data_path, args.train_time_steps, rank=0,
                            trajectory_keys=train_keys,
                            max_samples=args.max_train_samples)
    val_ds = DGPdeDataset(data_path, args.train_time_steps, rank=0,
                          trajectory_keys=val_keys)
    train_loader = create_dg_loader(train_ds, batch_size=args.batch_size,
                                    shuffle=True, num_workers=2)
    val_loader = create_dg_loader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2)

    results = []
    for name in args.configs:
        if name not in ABLATION_CONFIGS:
            print(f"[skip] unknown config '{name}'")
            continue
        results.append(train_eval_one(name, args, device, train_loader, val_loader))

    # reference = full (if present)
    ref = next((r for r in results if r["config"] == "full"), results[0])
    ref_err = ref["best_val_rel_err"]
    for r in results:
        r["delta_vs_full"] = r["best_val_rel_err"] - ref_err

    # CSV
    csv_path = out_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # Table
    print("\n" + "=" * 74)
    print(f"{'config':<16}{'params':>12}{'val_loss':>12}{'val_relErr':>12}{'Δ vs full':>12}")
    print("-" * 74)
    for r in results:
        print(f"{r['config']:<16}{r['params']:>12,}{r['best_val_loss']:>12.5f}"
              f"{r['best_val_rel_err']:>12.4f}{r['delta_vs_full']:>+12.4f}")
    print("=" * 74)
    print("Interpretation: a LARGER positive Δ means removing that component HURT "
          "accuracy more,\ni.e. that component contributes more. (static_anchor "
          "isolates the dynamic-update gain.)")

    # Bar chart
    order = [r["config"] for r in results]
    errs = [r["best_val_rel_err"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2c7fb8" if c == "full" else "#de2d26" if c == "baseline" else "#7fbf7b"
              for c in order]
    bars = ax.bar(order, errs, color=colors)
    if not np.isnan(ref_err):
        ax.axhline(ref_err, color="#2c7fb8", ls="--", lw=1, label="full")
    ax.set_ylabel("validation rel L2 error (final step)")
    ax.set_title(f"PhysHGNet ablation (N={args.n_nodes}, {args.epochs} epochs)")
    ax.tick_params(axis="x", rotation=20)
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, e, f"{e:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_bar.png", dpi=140)
    plt.close(fig)
    print(f"\n[csv]  {csv_path}\n[plot] {out_dir/'ablation_bar.png'}")


if __name__ == "__main__":
    main()
