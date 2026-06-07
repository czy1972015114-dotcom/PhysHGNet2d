"""
benchmark_inference.py — 实验 2.5：在「未见过的轨迹」上对比
PhysHGNet 与 DGNet 的推理误差（MSE / RNE）、推理速度、推理显存占用。

做法：
  · 加载在 N=4000 上训练好的两个模型的 best checkpoint；
  · 在一个独立生成的测试集（不同随机种子的轨迹）上做整段 rollout 推理；
  · 速度：每条轨迹 forward 的 wall-clock（含 torch.cuda.synchronize，预热若干次后计时）；
  · 显存：torch.cuda.max_memory_allocated（每个模型单独 reset 后测量）；
  · 误差：整段 rollout（t=1..T-1）的 MSE / RNE。

用法：
    python experiments/benchmark_inference.py \
        --phys-ckpt checkpoints/phys_hgnet/best_4000.pth \
        --dgnet-ckpt checkpoints/dgnet/best_4000.pth \
        --data-path data_laser_hardening_test/pde_trajectories_4000.h5 \
        --out results/benchmark_inference_4000.json
"""

import os
import sys
import json
import time
import argparse
import pathlib

import torch
import h5py

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dataset import DGPdeDataset, create_dg_loader   # noqa: E402
from metrics import mse as mse_metric, rne as rne_metric   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phys-ckpt", type=str, required=True)
    p.add_argument("--dgnet-ckpt", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True,
                   help="未见轨迹的测试集 HDF5（建议用不同 seed 生成）")
    p.add_argument("--train-time-steps", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


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


def load_model(kind, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    if kind == "physhgnet":
        from phys_hgnet import PhysHGNet
        model = PhysHGNet(config)
    else:
        from dgnet import DGNet
        model = DGNet(config)
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def benchmark(model, batches, device, warmup=2):
    cuda = device.type == "cuda"
    # 预热（含 LU/图缓存的首次构建，不计时）
    for i in range(min(warmup, len(batches))):
        _ = model(batches[i])
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    mse_sum = rne_sum = 0.0
    t_sum = 0.0
    n = 0
    for batch in batches:
        if cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model(batch)["u_final"]
        if cuda:
            torch.cuda.synchronize(device)
        t_sum += time.perf_counter() - t0

        tgt = batch["node_features"]
        mse_sum += mse_metric(out[:, 1:], tgt[:, 1:])
        rne_sum += rne_metric(out[:, 1:], tgt[:, 1:])
        n += 1

    peak_mem = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if cuda else float("nan")
    return {
        "mse": mse_sum / max(n, 1),
        "rne": rne_sum / max(n, 1),
        "time_per_traj_s": t_sum / max(n, 1),
        "peak_mem_MB": peak_mem,
        "num_traj": n,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = pathlib.Path(args.data_path)
    if not data_path.is_absolute():
        data_path = _REPO / args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"测试集不存在: {data_path}")

    with h5py.File(data_path, "r") as f:
        keys = sorted(f.keys())
    ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                      trajectory_keys=keys)
    loader = create_dg_loader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, sampler=None)
    # 物化到内存，保证两模型用完全相同的输入与顺序
    batches = [_to_device(b, device) for b in loader]

    results = {}
    for kind, ckpt in [("physhgnet", args.phys_ckpt), ("dgnet", args.dgnet_ckpt)]:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            model = load_model(kind, ckpt, device)
            r = benchmark(model, batches, device, warmup=args.warmup)
            del model
        except RuntimeError as e:
            msg = str(e).lower()
            r = {"error": "OOM" if "out of memory" in msg else "RuntimeError",
                 "detail": str(e)[:300]}
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[kind] = r
        print(f"\n[{kind}] {json.dumps(r, ensure_ascii=False)}")

    # 简表
    print("\n==================== Inference Benchmark (2.5) ====================")
    print(f"{'model':<12}{'MSE':>14}{'RNE':>12}{'time/traj(s)':>16}{'peak mem(MB)':>16}")
    for k, r in results.items():
        if "error" in r:
            print(f"{k:<12}{r['error']:>14}")
        else:
            print(f"{k:<12}{r['mse']:>14.6e}{r['rne']:>12.4f}"
                  f"{r['time_per_traj_s']:>16.4f}{r['peak_mem_MB']:>16.1f}")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[benchmark] 已写入 {args.out}")


if __name__ == "__main__":
    main()
