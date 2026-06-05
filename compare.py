"""
compare.py — Unified comparison between DGNet and PhysHGNet.

Loads trained checkpoints for both models and evaluates them on the full
validation set, reporting:
  - MSE  (Mean Squared Error on predicted trajectories)
  - RNE  (Relative Norm Error = ||pred - gt|| / ||gt||)
  - Peak GPU memory (MiB) during inference
  - Scaling experiment: MSE/memory vs number of anchor points

Usage:
    python compare.py                  # compare on all val trajectories
    python compare.py --scaling        # also run scaling experiment
    python compare.py --no-plot        # skip matplotlib plots
    # 只指定节点数（自动定位 best_{n_nodes}.pth）：
python compare.py --n-nodes 2000
# 或显式指定路径：
python compare.py --dgnet-ckpt checkpoints/dgnet/best_2000.pth \\
                  --physhgnet-ckpt checkpoints/phys_hgnet/best_2000.pth

Outputs:
    results/comparison_results.json   — all metrics
    results/figures/                  — trajectory and metric plots
"""

import argparse
import json
import os
import pathlib
import time
import warnings

import torch
import numpy as np
import h5py

from dataset import DGPdeDataset, create_dg_loader


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MSE across all elements."""
    with torch.no_grad():
        return torch.mean((pred - target) ** 2).item()


def compute_rne(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Norm Error: ||pred - target|| / ||target||."""
    with torch.no_grad():
        return (torch.norm(pred - target) /
                torch.norm(target).clamp(min=1e-8)).item()


def compute_per_step_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """MSE at each time step: returns array of shape (T,)."""
    with torch.no_grad():
        T = pred.shape[1]
        return np.array([
            torch.mean((pred[:, t] - target[:, t]) ** 2).item()
            for t in range(T)
        ])


# ─────────────────────────────────────────────────────────────
# Inference with memory measurement
# ─────────────────────────────────────────────────────────────

def run_inference(model, loader, device, model_name="model"):
    """Run model inference on the full loader, return metrics."""
    model.eval()
    all_mse, all_rne = [], []
    all_per_step_mse = []
    peak_mem_mib = 0.0
    total_time = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            t0 = time.time()
            out = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            total_time += elapsed

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_mem_mib = max(peak_mem_mib, mem)

            pred = out["u_final"]          # (B, T, N, C)
            target = batch["node_features"]  # (B, T, N, C)

            all_mse.append(compute_mse(pred[:, -1], target[:, -1]))
            all_rne.append(compute_rne(pred[:, -1], target[:, -1]))
            all_per_step_mse.append(compute_per_step_mse(pred, target))

    metrics = {
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "rne_mean": float(np.mean(all_rne)),
        "rne_std": float(np.std(all_rne)),
        "peak_gpu_mem_mib": float(peak_mem_mib),
        "total_inference_time_s": float(total_time),
        "per_step_mse": np.mean(all_per_step_mse, axis=0).tolist(),
    }
    return metrics


def _to_device(batch, device):
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


# ─────────────────────────────────────────────────────────────
# Scaling experiment: vary m_anchors for PhysHGNet
# ─────────────────────────────────────────────────────────────

def scaling_experiment(physhgnet_ckpt_path, data_path, device,
                       anchor_list=(32, 64, 128, 256)):
    """
    Evaluate PhysHGNet with different numbers of anchor points (m_anchors).
    This tests the memory-vs-accuracy tradeoff of the hierarchical design.
    """
    from phys_hgnet import PhysHGNet
    results = {}

    # Load base config
    ckpt = torch.load(physhgnet_ckpt_path, map_location=device)
    base_config = ckpt["config"]

    # Validation data
    with h5py.File(data_path, 'r') as f:
        all_keys = sorted(list(f.keys()))
    split_idx = max(1, int(0.8 * len(all_keys)))
    val_keys = all_keys[split_idx:] or [all_keys[-1]]
    val_ds = DGPdeDataset(data_path, train_time_steps=7, trajectory_keys=val_keys)
    val_loader = create_dg_loader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    for m in anchor_list:
        config = dict(base_config)
        config["m_anchors"] = m
        model = PhysHGNet(config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        mse_list, mem_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                out = model(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_list.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                mse_list.append(compute_mse(out["u_final"][:, -1], batch["node_features"][:, -1]))

        results[m] = {
            "mse": float(np.mean(mse_list)),
            "peak_gpu_mem_mib": float(np.mean(mem_list)) if mem_list else 0.0,
        }
        print(f"  m_anchors={m:4d} | MSE={results[m]['mse']:.6f} | "
              f"Mem={results[m]['peak_gpu_mem_mib']:.1f} MiB")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def _save_plots(dgnet_metrics, physhgnet_metrics, scaling_results, figures_dir, n_nodes=""):
    """Save comparison figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Warning] matplotlib not available — skipping plots.")
        return

    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{n_nodes}" if n_nodes else ""

    # ── 1. Bar chart: MSE and RNE ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = ["DGNet", "PhysHGNet"]
    mse_vals = [dgnet_metrics["mse_mean"], physhgnet_metrics["mse_mean"]]
    mse_errs = [dgnet_metrics["mse_std"], physhgnet_metrics["mse_std"]]
    rne_vals = [dgnet_metrics["rne_mean"], physhgnet_metrics["rne_mean"]]
    rne_errs = [dgnet_metrics["rne_std"], physhgnet_metrics["rne_std"]]
    colors = ["#4C72B0", "#DD8452"]

    axes[0].bar(models, mse_vals, yerr=mse_errs, color=colors, capsize=5,
                edgecolor='black', linewidth=0.8)
    axes[0].set_title("MSE (Final Step)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("MSE")
    for i, (v, e) in enumerate(zip(mse_vals, mse_errs)):
        axes[0].text(i, v + e + max(mse_vals) * 0.01, f"{v:.2e}",
                     ha='center', va='bottom', fontsize=9)

    axes[1].bar(models, rne_vals, yerr=rne_errs, color=colors, capsize=5,
                edgecolor='black', linewidth=0.8)
    axes[1].set_title("RNE (Final Step, Relative Norm Error)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("RNE")
    for i, (v, e) in enumerate(zip(rne_vals, rne_errs)):
        axes[1].text(i, v + e + max(rne_vals) * 0.01, f"{v:.4f}",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / f"mse_rne_comparison{sfx}.pdf", bbox_inches='tight', dpi=150)
    plt.savefig(figures_dir / f"mse_rne_comparison{sfx}.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {figures_dir / f'mse_rne_comparison{sfx}.png'}")

    # ── 2. Per-step MSE curve ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    T = max(len(dgnet_metrics["per_step_mse"]), len(physhgnet_metrics["per_step_mse"]))
    steps = list(range(T))
    ax.plot(steps, dgnet_metrics["per_step_mse"][:T], label="DGNet",
            marker='o', color=colors[0], linewidth=1.5)
    ax.plot(steps, physhgnet_metrics["per_step_mse"][:T], label="PhysHGNet",
            marker='s', color=colors[1], linewidth=1.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs Time Step", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"per_step_mse{sfx}.pdf", bbox_inches='tight', dpi=150)
    plt.savefig(figures_dir / f"per_step_mse{sfx}.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {figures_dir / f'per_step_mse{sfx}.png'}")

    # ── 3. Memory bar chart ───────────────────────────────────
    if dgnet_metrics.get("peak_gpu_mem_mib", 0) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        mem_vals = [dgnet_metrics["peak_gpu_mem_mib"], physhgnet_metrics["peak_gpu_mem_mib"]]
        bars = ax.bar(models, mem_vals, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_title("Peak GPU Memory (MiB)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Memory (MiB)")
        for bar, v in zip(bars, mem_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{v:.0f} MiB", ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(figures_dir / f"memory_comparison{sfx}.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved: {figures_dir / f'memory_comparison{sfx}.png'}")

    # ── 4. Scaling plot (if available) ───────────────────────
    if scaling_results:
        anchors = sorted(scaling_results.keys())
        mse_sc = [scaling_results[m]["mse"] for m in anchors]
        mem_sc = [scaling_results[m]["peak_gpu_mem_mib"] for m in anchors]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ln1 = ax1.plot(anchors, mse_sc, 'o-', color=colors[1], label="MSE", linewidth=1.5)
        ln2 = ax2.plot(anchors, mem_sc, 's--', color='#55A868', label="GPU Mem (MiB)", linewidth=1.5)
        ax1.set_xlabel("Anchor Points (m)")
        ax1.set_ylabel("MSE")
        ax2.set_ylabel("Peak GPU Memory (MiB)")
        ax1.set_title("PhysHGNet Scaling: Accuracy vs Memory", fontsize=13, fontweight='bold')
        lns = ln1 + ln2
        ax1.legend(lns, [l.get_label() for l in lns], loc='upper right')
        ax1.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"scaling_experiment{sfx}.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved: {figures_dir / f'scaling_experiment{sfx}.png'}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgnet-ckpt", type=str, default=None,
                        help="DGNet 检查点路径；不传时按 --n-nodes 自动推断 "
                             "checkpoints/dgnet/best_{n_nodes}.pth")
    parser.add_argument("--physhgnet-ckpt", type=str, default=None,
                        help="PhysHGNet 检查点路径；不传时按 --n-nodes 自动推断 "
                             "checkpoints/phys_hgnet/best_{n_nodes}.pth")
    parser.add_argument("--n-nodes",   type=int, default=2000,
                        help="数据节点数，自动定位 pde_trajectories_{n_nodes}.h5")
    parser.add_argument("--data-dir",  type=str, default="data_laser_hardening",
                        help="数据目录")
    parser.add_argument("--data-path", type=str, default=None,
                        help="显式指定 HDF5 路径，传入后覆盖 --data-dir + --n-nodes")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--scaling", action="store_true",
                        help="Run PhysHGNet scaling experiment (vary m_anchors)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_dir = pathlib.Path(__file__).parent.resolve()
    # --data-path 显式传入时直接使用，否则按 --n-nodes 自动构造
    if args.data_path:
        data_path = (pathlib.Path(args.data_path) if pathlib.Path(args.data_path).is_absolute()
                     else base_dir / args.data_path)
    else:
        data_path = base_dir / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DGNet vs PhysHGNet — Comparison Experiment")
    print(f"  Device: {device}")
    print(f"  Data:   {data_path}")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}\n"
            f"Run: python generate_laser_data_aligned.py --n_nodes {args.n_nodes} --out_dir {args.data_dir}")

    # ── Validation dataset ───────────────────────────────────
    with h5py.File(data_path, 'r') as f:
        all_keys = sorted(list(f.keys()))
    split_idx = max(1, int(0.8 * len(all_keys)))
    val_keys = all_keys[split_idx:] or [all_keys[-1]]
    print(f"  Validation trajectories: {len(val_keys)}")

    val_ds = DGPdeDataset(data_path, train_time_steps=7, trajectory_keys=val_keys)
    val_loader = create_dg_loader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

    comparison_results = {}

    # ── Evaluate DGNet ───────────────────────────────────────
    # 按 --n-nodes 自动推断检查点路径（显式传入时直接使用）
    n_nodes = args.n_nodes
    dgnet_ckpt = (base_dir / args.dgnet_ckpt if args.dgnet_ckpt
                  else base_dir / "checkpoints" / "dgnet" / f"best_{n_nodes}.pth")
    if dgnet_ckpt.exists():
        print(f"\n[1/2] Loading DGNet from {dgnet_ckpt} ...")
        from dgnet import DGNet
        ckpt = torch.load(dgnet_ckpt, map_location=device)
        dgnet_config = ckpt["config"]
        dgnet_model = DGNet(dgnet_config).to(device)
        dgnet_model.load_state_dict(ckpt["model_state_dict"])
        print("  Running inference ...")
        dgnet_metrics = run_inference(dgnet_model, val_loader, device, "DGNet")
        comparison_results["DGNet"] = dgnet_metrics
        print(f"  DGNet Results:")
        print(f"    MSE          = {dgnet_metrics['mse_mean']:.6e} ± {dgnet_metrics['mse_std']:.2e}")
        print(f"    RNE          = {dgnet_metrics['rne_mean']:.4f} ± {dgnet_metrics['rne_std']:.4f}")
        print(f"    Peak GPU Mem = {dgnet_metrics['peak_gpu_mem_mib']:.1f} MiB")
        print(f"    Inference t  = {dgnet_metrics['total_inference_time_s']:.2f} s")
        del dgnet_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"  [SKIP] DGNet checkpoint not found: {dgnet_ckpt}")
        dgnet_metrics = None

    # ── Evaluate PhysHGNet ───────────────────────────────────
    physhgnet_ckpt = (base_dir / args.physhgnet_ckpt if args.physhgnet_ckpt
                      else base_dir / "checkpoints" / "phys_hgnet" / f"best_{n_nodes}.pth")
    if physhgnet_ckpt.exists():
        print(f"\n[2/2] Loading PhysHGNet from {physhgnet_ckpt} ...")
        from phys_hgnet import PhysHGNet
        ckpt = torch.load(physhgnet_ckpt, map_location=device)
        physhgnet_config = ckpt["config"]
        physhgnet_model = PhysHGNet(physhgnet_config).to(device)
        physhgnet_model.load_state_dict(ckpt["model_state"])
        print(physhgnet_model.ablation_summary())
        print("  Running inference ...")
        physhgnet_metrics = run_inference(physhgnet_model, val_loader, device, "PhysHGNet")
        comparison_results["PhysHGNet"] = physhgnet_metrics
        print(f"  PhysHGNet Results:")
        print(f"    MSE          = {physhgnet_metrics['mse_mean']:.6e} ± {physhgnet_metrics['mse_std']:.2e}")
        print(f"    RNE          = {physhgnet_metrics['rne_mean']:.4f} ± {physhgnet_metrics['rne_std']:.4f}")
        print(f"    Peak GPU Mem = {physhgnet_metrics['peak_gpu_mem_mib']:.1f} MiB")
        print(f"    Inference t  = {physhgnet_metrics['total_inference_time_s']:.2f} s")
        del physhgnet_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"  [SKIP] PhysHGNet checkpoint not found: {physhgnet_ckpt}")
        physhgnet_metrics = None

    # ── Summary table ────────────────────────────────────────
    if dgnet_metrics and physhgnet_metrics:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<30} {'DGNet':>14} {'PhysHGNet':>14} {'Δ (%)':>10}")
        print("-" * 68)

        def diff_pct(a, b):
            return f"{(b - a) / max(abs(a), 1e-12) * 100:+.1f}%"

        for metric, key_mean in [("MSE (final step)", "mse_mean"),
                                  ("RNE (final step)", "rne_mean"),
                                  ("Peak GPU Mem (MiB)", "peak_gpu_mem_mib")]:
            dg = dgnet_metrics[key_mean]
            ph = physhgnet_metrics[key_mean]
            print(f"{metric:<30} {dg:>14.4e} {ph:>14.4e} {diff_pct(dg, ph):>10}")

        print("=" * 60)

        improvement_mse = (dgnet_metrics["mse_mean"] - physhgnet_metrics["mse_mean"]) \
                          / dgnet_metrics["mse_mean"] * 100
        improvement_rne = (dgnet_metrics["rne_mean"] - physhgnet_metrics["rne_mean"]) \
                          / dgnet_metrics["rne_mean"] * 100
        mem_saving = (dgnet_metrics["peak_gpu_mem_mib"] - physhgnet_metrics["peak_gpu_mem_mib"]) \
                     / max(dgnet_metrics["peak_gpu_mem_mib"], 1) * 100

        print(f"\nPhysHGNet vs DGNet:")
        print(f"  MSE reduction:    {improvement_mse:+.1f}%")
        print(f"  RNE reduction:    {improvement_rne:+.1f}%")
        print(f"  Memory reduction: {mem_saving:+.1f}%")

    # ── Scaling experiment ───────────────────────────────────
    scaling_results = {}
    if args.scaling and physhgnet_ckpt.exists():
        print("\n[Scaling Experiment] PhysHGNet m_anchors vs MSE/Memory ...")
        scaling_results = scaling_experiment(
            str(physhgnet_ckpt), str(data_path), device,
            anchor_list=[16, 32, 64, 128, 256])
        comparison_results["scaling"] = scaling_results

    # ── Save results ─────────────────────────────────────────
    save_path = results_dir / f"comparison_results_{n_nodes}.json"
    with open(save_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    # ── Plots ────────────────────────────────────────────────
    if not args.no_plot and dgnet_metrics and physhgnet_metrics:
        print("\nGenerating figures ...")
        _save_plots(dgnet_metrics, physhgnet_metrics, scaling_results, figures_dir, n_nodes)


if __name__ == "__main__":
    main()

