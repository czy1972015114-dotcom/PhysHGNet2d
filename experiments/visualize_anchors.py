"""
visualize_anchors.py — 实验 2.6：锚点位置随时间变化的可视化。

加载 N=4000 训练好的 PhysHGNet 最佳模型，在【一条轨迹】上做推理，
取出每个时间步实际生效的锚点（forward(..., return_anchor_history=True)），
把锚点位置叠加在【温度场】上随时间绘制，用于观察“锚点分布 ↔ 温度”的关系。

产出（默认写入 results/）：
    1. anchor_vis_{N}_snapshots.png  —— 多个时间步的快照拼图（温度等值填色 + 锚点散点）
    2. anchor_vis_{N}.gif            —— 锚点随时间移动的动画（若 pillow 可用）
    3. anchor_vs_temperature_{N}.png —— 定量曲线：落在高温区的锚点占比 vs 时间

说明：
  · 单卡推理即可，无需 DDP。
  · 为了得到更长的时间演化，本脚本直接从 HDF5 读取一条完整轨迹并按 --stride
    下采样出 T_vis 帧，而不是走 dataset 的 7 步分块。
  · 温度场默认用【真值】node_features（物理上的真实温度，激光热源驱动），
    叠加的锚点则来自模型 rollout 时内部状态的实时选择；
    第 t 帧的锚点 anchor_history[t] 对应的是模型在第 t 帧状态上选出的锚点，
    因此叠加在温度真值的第 t 帧最为自洽。
    若想叠加在模型预测温度上，加 --use-pred。
"""

import os
import sys
import json
import argparse
import pathlib

import numpy as np
import torch
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="PhysHGNet 最佳 checkpoint，如 checkpoints/phys_hgnet/best_4000.pth")
    p.add_argument("--n-nodes", type=int, default=4000)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data_laser_hardening")
    p.add_argument("--traj-index", type=int, default=0, help="使用第几条轨迹（按排序后的 key）")
    p.add_argument("--time-window", type=int, default=60,
                   help="可视化的帧数（从轨迹起点开始，配合 --stride）")
    p.add_argument("--stride", type=int, default=2, help="时间下采样步长")
    p.add_argument("--coarse-update-freq", type=int, default=1,
                   help="锚点刷新频率；2.6 建议设为 1 使锚点逐帧更新、运动更平滑")
    p.add_argument("--n-snapshots", type=int, default=6, help="快照拼图的面板数")
    p.add_argument("--hot-percentile", type=float, default=80.0,
                   help="高温区阈值分位数（默认温度前 20%% 记为高温区）")
    p.add_argument("--use-pred", action="store_true",
                   help="温度底图改用模型预测场（默认用真值 node_features）")
    p.add_argument("--no-gif", action="store_true", help="跳过 GIF 生成")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def resolve_data_path(args):
    if args.data_path:
        p = pathlib.Path(args.data_path)
        return p if p.is_absolute() else _REPO / args.data_path
    return _REPO / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"


def load_single_trajectory(data_path, traj_index, time_window, stride):
    """直接从 HDF5 读取一条轨迹，构造单样本 batch（[1,T,N,C]）。"""
    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
        if traj_index >= len(keys):
            raise IndexError(f"traj-index={traj_index} 超出范围（共 {len(keys)} 条轨迹）")
        g = f[keys[traj_index]]
        nodes = np.asarray(g["nodes"])               # [N,2]
        edges = np.asarray(g["edges"])               # [E,2]
        faces = np.asarray(g["faces"])               # [F,3]
        node_feat = np.asarray(g["node_features"])   # [T,N,C]
        src = np.asarray(g["source_terms"])          # [T,N,C]
        tpts = np.asarray(g["time_points"])          # [T]
        if "node_volumes" in g:
            node_vol = np.asarray(g["node_volumes"])
        else:
            node_vol = None
        if "node_type" in g:
            node_type = np.asarray(g["node_type"])
        else:
            node_type = np.zeros(nodes.shape[0], dtype=np.int64)

    T_full = node_feat.shape[0]
    idx = list(range(0, min(time_window * stride, T_full), stride))
    if len(idx) < 2:
        idx = list(range(min(2, T_full)))
    node_feat = node_feat[idx]
    src = src[idx]
    tpts = tpts[idx]

    def tf(x, dt=torch.float32):
        return torch.as_tensor(np.asarray(x)).to(dt)

    batch = {
        "nodes": tf(nodes),
        "edges": torch.as_tensor(np.asarray(edges)).long(),
        "faces": torch.as_tensor(np.asarray(faces)).long(),
        "node_features": tf(node_feat).unsqueeze(0),     # [1,T,N,C]
        "source_terms": tf(src).unsqueeze(0),            # [1,T,N,C]
        "initial_conditions": tf(node_feat[0]).unsqueeze(0),  # [1,N,C]
        "time_points": tf(tpts),
        "node_type": torch.as_tensor(node_type).long(),
        "node_volumes": (tf(node_vol) if node_vol is not None else None),
        "boundary_info": {},
    }
    return batch


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


def build_model(config, coarse_update_freq):
    from phys_hgnet import PhysHGNet
    cfg = dict(config) if config else {}
    cfg["coarse_update_freq"] = coarse_update_freq   # 2.6 建议逐帧刷新
    return PhysHGNet(cfg)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt.get("config", {})
    model = build_model(config, args.coarse_update_freq).to(device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    model.load_state_dict(state)
    model.eval()

    batch = load_single_trajectory(data_path, args.traj_index,
                                   args.time_window, args.stride)
    batch = to_device(batch, device)

    with torch.no_grad():
        out = model(batch, return_anchor_history=True)
    u_pred = out["u_final"][0, :, :, 0].detach().cpu().numpy()   # [T,N]
    anchor_hist = out["anchor_history"]                          # list[T-1] of LongTensor[m]
    nodes = out.get("nodes")
    if nodes is None:
        nodes = batch["nodes"].detach().cpu()
    nodes = np.asarray(nodes)                                    # [N,2]
    u_true = batch["node_features"][0, :, :, 0].detach().cpu().numpy()  # [T,N]
    faces = batch["faces"].detach().cpu().numpy()

    temp = u_pred if args.use_pred else u_true                   # 底图温度场 [T,N]
    T = temp.shape[0]
    n_anchor_frames = len(anchor_hist)                           # = 模型 rollout 的 T-1

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], faces)
    vmin, vmax = float(np.min(temp)), float(np.max(temp))

    # ── 1. 快照拼图 ───────────────────────────────────────────
    n_show = min(args.n_snapshots, n_anchor_frames)
    frame_ids = np.linspace(0, n_anchor_frames - 1, n_show).astype(int)
    ncol = min(3, n_show)
    nrow = int(np.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow),
                             squeeze=False)
    last_tcf = None
    for i, fid in enumerate(frame_ids):
        ax = axes[i // ncol][i % ncol]
        tcf = ax.tricontourf(triang, temp[fid], levels=24, cmap="inferno",
                             vmin=vmin, vmax=vmax)
        last_tcf = tcf
        a_idx = anchor_hist[fid].numpy()
        ax.scatter(nodes[a_idx, 0], nodes[a_idx, 1], s=14,
                   facecolors="none", edgecolors="cyan", linewidths=0.9,
                   label="anchors")
        ax.set_title(f"frame t={fid}  (m={len(a_idx)})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n_show, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"PhysHGNet 锚点位置 vs 温度场  (N={args.n_nodes}, "
                 f"{'pred' if args.use_pred else 'true'} temperature)",
                 fontsize=12)
    if last_tcf is not None:
        cbar = fig.colorbar(last_tcf, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("temperature")
    snap_path = out_dir / f"anchor_vis_{args.n_nodes}_snapshots.png"
    fig.savefig(snap_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[2.6] 快照拼图已保存: {snap_path}")

    # ── 2. 定量曲线：高温区锚点占比 vs 时间 ────────────────────
    frac_in_hot = []
    mean_anchor_temp = []
    mean_field_temp = []
    for fid in range(n_anchor_frames):
        a_idx = anchor_hist[fid].numpy()
        thr = np.percentile(temp[fid], args.hot_percentile)
        in_hot = np.mean(temp[fid][a_idx] >= thr) if len(a_idx) else 0.0
        frac_in_hot.append(float(in_hot))
        mean_anchor_temp.append(float(np.mean(temp[fid][a_idx])) if len(a_idx) else 0.0)
        mean_field_temp.append(float(np.mean(temp[fid])))

    fig2, ax1 = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(n_anchor_frames)
    ax1.plot(xs, frac_in_hot, "o-", color="tab:red",
             label=f"锚点落在温度前 {100 - args.hot_percentile:.0f}% 区域的占比")
    ax1.axhline(1.0 - args.hot_percentile / 100.0, ls="--", color="gray",
                lw=1, label="随机均匀分布期望值")
    ax1.set_xlabel("time frame")
    ax1.set_ylabel("fraction in hot region")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(xs, mean_anchor_temp, "s-", color="tab:blue", alpha=0.7,
             label="锚点处平均温度")
    ax2.plot(xs, mean_field_temp, "--", color="tab:green", alpha=0.7,
             label="全场平均温度")
    ax2.set_ylabel("temperature")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.set_title(f"锚点分布与温度的关系 (N={args.n_nodes})")
    quant_path = out_dir / f"anchor_vs_temperature_{args.n_nodes}.png"
    fig2.savefig(quant_path, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"[2.6] 定量曲线已保存: {quant_path}")

    # 同时把定量数据写出 JSON，便于复用
    quant_json = out_dir / f"anchor_vs_temperature_{args.n_nodes}.json"
    with open(quant_json, "w", encoding="utf-8") as fp:
        json.dump({
            "n_nodes": args.n_nodes,
            "hot_percentile": args.hot_percentile,
            "frac_in_hot": frac_in_hot,
            "mean_anchor_temp": mean_anchor_temp,
            "mean_field_temp": mean_field_temp,
        }, fp, indent=2, ensure_ascii=False)
    print(f"[2.6] 定量数据已保存: {quant_json}")

    # ── 3. GIF 动画 ───────────────────────────────────────────
    if not args.no_gif:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            figg, axg = plt.subplots(figsize=(5.4, 4.8))

            def draw(fid):
                axg.clear()
                axg.tricontourf(triang, temp[fid], levels=24, cmap="inferno",
                                vmin=vmin, vmax=vmax)
                a_idx = anchor_hist[fid].numpy()
                axg.scatter(nodes[a_idx, 0], nodes[a_idx, 1], s=16,
                            facecolors="none", edgecolors="cyan", linewidths=1.0)
                axg.set_title(f"PhysHGNet anchors  t={fid}  (N={args.n_nodes})",
                              fontsize=10)
                axg.set_aspect("equal")
                axg.set_xticks([]); axg.set_yticks([])

            anim = FuncAnimation(figg, draw, frames=n_anchor_frames, interval=200)
            gif_path = out_dir / f"anchor_vis_{args.n_nodes}.gif"
            anim.save(gif_path, writer=PillowWriter(fps=5))
            plt.close(figg)
            print(f"[2.6] 动画已保存: {gif_path}")
        except Exception as e:
            print(f"[2.6] GIF 生成跳过（{type(e).__name__}: {e}）"
                  f"；快照与定量图已生成。")

    print("[2.6] 完成。")


if __name__ == "__main__":
    main()
