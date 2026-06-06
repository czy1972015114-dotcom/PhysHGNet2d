"""
visualize_anchors.py — Visualise the 2D temperature field together with the
PhysHGNet anchor points, and show how the anchors move as the rollout advances.

Outputs (under --out-dir):
  frames/frame_000.png ... frame_TTT.png   per-timestep temperature + anchors
  anchor_evolution.gif                      animated GIF of the whole rollout
  screenshot_t000.png / _tmid.png / _tend.png   three high-res still screenshots
  anchor_evolution_montage.png              small-multiples montage (6 timesteps)
  anchor_count.png                          #anchors actually selected per update

What it demonstrates
--------------------
With the corrected phys_hgnet.py the anchor set is re-selected every
`--update-freq` steps from the live PDE residual + temperature gradient, so the
anchors visibly migrate to follow the moving laser hot-spots. Set
`--update-freq 1` for the smoothest animation.

Usage
-----
  # Auto-locate data_laser_hardening/pde_trajectories_2000.h5
  python visualize_anchors.py --n-nodes 2000

  # With a trained checkpoint and a specific trajectory
  python visualize_anchors.py --n-nodes 2000 \
      --ckpt checkpoints/phys_hgnet/best_2000.pth \
      --traj-index 0 --update-freq 1 --out-dir viz_anchors
"""
import os
import argparse
import pathlib
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from phys_hgnet import PhysHGNet, DEFAULT_CONFIG


# ──────────────────────────────────────────────────────────────
# Data loading: read ONE full-length trajectory directly from HDF5
# ──────────────────────────────────────────────────────────────
def load_trajectory(data_path, traj_index, device):
    import h5py
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
        key = keys[traj_index] if isinstance(traj_index, int) else traj_index
        g = f[key]
        nodes = torch.from_numpy(g["nodes"][:]).float()
        edges = torch.from_numpy(g["edges"][:]).long()
        faces = torch.from_numpy(g["faces"][:]).long()
        node_features = torch.from_numpy(g["node_features"][:]).float()   # [T, N, 1]
        source_terms = torch.from_numpy(g["source_terms"][:]).float()     # [T, N, 1]
        time_points = torch.from_numpy(g["time_points"][:]).float()       # [T]
        node_type = (torch.from_numpy(g["node_type"][:]).long()
                     if "node_type" in g else torch.zeros(nodes.shape[0], dtype=torch.long))
        node_volumes = (torch.from_numpy(g["node_volumes"][:]).float()
                        if "node_volumes" in g else None)
    T = node_features.shape[0]
    batch = {
        "nodes": nodes.to(device),
        "edges": edges.to(device),
        "faces": faces.to(device),
        "node_type": node_type.to(device),
        "node_volumes": (node_volumes.to(device) if node_volumes is not None else None),
        "boundary_info": {},
        "source_terms": source_terms.unsqueeze(0).to(device),        # [1, T, N, 1]
        "time_points": time_points.to(device),                       # [T]
        "initial_conditions": node_features[0].unsqueeze(0).to(device),  # [1, N, 1]
        "node_features": node_features.unsqueeze(0).to(device),      # [1, T, N, 1] (reference)
    }
    return batch, key, T


def build_model(args, device):
    cfg = dict(DEFAULT_CONFIG)
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg.update(ckpt["config"])
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    else:
        state = None
        if args.ckpt:
            print(f"[warn] checkpoint not found: {args.ckpt} — using random weights "
                  f"(temperature will be unphysical, anchor motion still illustrative)")

    # Force visualisation-friendly anchor behaviour (does not change param shapes)
    cfg["use_physics_anchor"] = True
    cfg["dynamic_anchors"] = True
    cfg["residual_update_freq"] = max(1, int(args.update_freq))
    cfg["m_anchors"] = args.m_anchors

    model = PhysHGNet(cfg).to(device)
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model


def anchors_at(history, t):
    """Most recently selected anchor set at or before step t."""
    chosen = history[0]["idx"]
    for h in history:
        if h["t"] <= t:
            chosen = h["idx"]
        else:
            break
    return chosen.numpy()


def plot_frame(ax, triang, temp, anchor_pts_xy, vmin, vmax, title):
    ax.clear()
    tcf = ax.tricontourf(triang, temp, levels=40, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.scatter(anchor_pts_xy[:, 0], anchor_pts_xy[:, 1],
               s=42, facecolors="cyan", edgecolors="black", linewidths=0.7,
               zorder=5, label=f"anchors (m={len(anchor_pts_xy)})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    return tcf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default="data_laser_hardening")
    ap.add_argument("--n-nodes", type=int, default=2000)
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--m-anchors", type=int, default=64)
    ap.add_argument("--update-freq", type=int, default=1,
                    help="re-select anchors every N steps (1 = every step, smoothest GIF)")
    ap.add_argument("--max-steps", type=int, default=80,
                    help="cap number of rendered frames (full trajectory can be 121)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="viz_anchors")
    args = ap.parse_args()

    base = pathlib.Path(__file__).parent.resolve()
    if args.data_path:
        data_path = pathlib.Path(args.data_path)
        if not data_path.is_absolute():
            data_path = base / data_path
    else:
        data_path = base / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}\n"
            f"Generate it first, e.g.:\n"
            f"  python generate_laser_data_aligned.py --n_nodes {args.n_nodes} "
            f"--out_dir {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = base / args.out_dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {data_path} (traj #{args.traj_index}) on {device}")
    batch, key, T = load_trajectory(data_path, args.traj_index, device)
    model = build_model(args, device)

    with torch.no_grad():
        out = model(batch, record_anchors=True)
    u_hist = out["u_final"][0, :, :, 0].cpu().numpy()   # [T, N]
    history = model.anchor_history
    print(f"[rollout] T={T}, anchor updates recorded={len(history)}")

    nodes = batch["nodes"].cpu().numpy()
    faces = batch["faces"].cpu().numpy()
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=faces)

    n_steps = min(T, args.max_steps)
    vmin = float(np.percentile(u_hist[:n_steps], 1))
    vmax = float(np.percentile(u_hist[:n_steps], 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    tvals = batch["time_points"].cpu().numpy()

    # ── per-step frames ──────────────────────────────────────
    frame_paths = []
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cbar = None
    for t in range(n_steps):
        anc = anchors_at(history, t)
        tcf = plot_frame(ax, triang, u_hist[t], nodes[anc], vmin, vmax,
                         title=f"traj {key} | step {t}/{T-1} | t={tvals[t]:.1f}s | ΔT field + anchors")
        if cbar is None:
            cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04, label="ΔT (K)")
        p = frames_dir / f"frame_{t:03d}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        frame_paths.append(p)
    plt.close(fig)
    print(f"[frames] wrote {len(frame_paths)} PNGs -> {frames_dir}")

    # ── three high-res screenshots ───────────────────────────
    for tag, t in [("t000", 0), ("tmid", n_steps // 2), ("tend", n_steps - 1)]:
        f2, a2 = plt.subplots(figsize=(7.2, 6.2))
        anc = anchors_at(history, t)
        tcf = plot_frame(a2, triang, u_hist[t], nodes[anc], vmin, vmax,
                         title=f"ΔT field + anchors @ step {t} (t={tvals[t]:.1f}s)")
        f2.colorbar(tcf, ax=a2, fraction=0.046, pad=0.04, label="ΔT (K)")
        f2.savefig(out_dir / f"screenshot_{tag}.png", dpi=160, bbox_inches="tight")
        plt.close(f2)
    print(f"[screenshots] screenshot_t000 / _tmid / _tend written")

    # ── evolution montage (6 timesteps) ──────────────────────
    picks = sorted(set(int(round(x)) for x in np.linspace(0, n_steps - 1, 6)))
    fm, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax_i, t in zip(axes.ravel(), picks):
        anc = anchors_at(history, t)
        plot_frame(ax_i, triang, u_hist[t], nodes[anc], vmin, vmax,
                   title=f"step {t} (t={tvals[t]:.1f}s)")
    for ax_i in axes.ravel()[len(picks):]:
        ax_i.axis("off")
    fm.suptitle("PhysHGNet anchor migration over time (cyan = anchors)", fontsize=14)
    fm.tight_layout(rect=[0, 0, 1, 0.97])
    fm.savefig(out_dir / "anchor_evolution_montage.png", dpi=130, bbox_inches="tight")
    plt.close(fm)
    print(f"[montage] anchor_evolution_montage.png written")

    # ── #anchors per update ──────────────────────────────────
    fc, axc = plt.subplots(figsize=(6, 3.4))
    ts = [h["t"] for h in history]
    ms = [len(h["idx"]) for h in history]
    axc.step(ts, ms, where="post", marker="o")
    axc.set_xlabel("time step"); axc.set_ylabel("# anchors selected")
    axc.set_title("Anchor count per re-selection")
    fc.tight_layout(); fc.savefig(out_dir / "anchor_count.png", dpi=120); plt.close(fc)

    # ── assemble GIF ─────────────────────────────────────────
    gif_path = out_dir / "anchor_evolution.gif"
    duration = 1.0 / max(1, args.fps)
    wrote = False
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, imgs, duration=duration, loop=0)
        wrote = True
    except Exception as e1:
        try:
            from PIL import Image
            imgs = [Image.open(p).convert("RGB") for p in frame_paths]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         duration=int(duration * 1000), loop=0)
            wrote = True
        except Exception as e2:
            print(f"[gif][warn] could not build GIF (imageio: {e1}; PIL: {e2}). "
                  f"Frames are still in {frames_dir}.")
    if wrote:
        print(f"[gif] {gif_path}")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
