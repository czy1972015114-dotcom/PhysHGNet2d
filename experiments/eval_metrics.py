"""
eval_metrics.py — 加载训练好的 checkpoint，在指定数据集上计算 MSE / RNE。

被 2.1 / 2.3 / 2.4 复用。单卡评测即可（评测无需 DDP）。

用法示例：
    python experiments/eval_metrics.py --model physhgnet \
        --ckpt checkpoints/phys_hgnet/best_2000.pth \
        --n-nodes 2000 --split val --out results/eval_phys_2000.json

    python experiments/eval_metrics.py --model dgnet \
        --ckpt checkpoints/dgnet/best_2000.pth \
        --n-nodes 2000 --split val

指标定义（见 metrics.py）：
    MSE = mean((pred-target)^2)            在 t=1..T-1 的整段 rollout 上
    RNE = ||pred-target||_2/||target||_2   同上
另外附带 final-step（仅最后一帧）的 MSE/RNE，便于和训练日志对照。
"""

import os
import sys
import json
import argparse
import pathlib

import torch
import h5py

# 允许从仓库根目录或 experiments/ 下运行
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dataset import DGPdeDataset, create_dg_loader   # noqa: E402
from metrics import mse as mse_metric, rne as rne_metric   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["physhgnet", "dgnet"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-nodes", type=int, default=2000)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data_laser_hardening")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--train-time-steps", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default=None, help="JSON 输出路径（可选）")
    return p.parse_args()


def build_model(model_name, config):
    if model_name == "physhgnet":
        from phys_hgnet import PhysHGNet
        return PhysHGNet(config)
    else:
        from dgnet import DGNet
        return DGNet(config)


def resolve_data_path(args):
    if args.data_path:
        p = pathlib.Path(args.data_path)
        return p if p.is_absolute() else _REPO / args.data_path
    return _REPO / args.data_dir / f"pde_trajectories_{args.n_nodes}.h5"


def pick_keys(data_path, split):
    with h5py.File(data_path, "r") as f:
        all_keys = sorted(f.keys())
    if split == "all":
        return all_keys
    s = max(1, int(0.8 * len(all_keys)))
    train_keys = all_keys[:s]
    val_keys = all_keys[s:] or [train_keys[-1]]
    return train_keys if split == "train" else val_keys


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = {"mse": 0.0, "rne": 0.0, "mse_last": 0.0, "rne_last": 0.0, "n": 0}
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)["u_final"]            # [B,T,N,C]
        tgt = batch["node_features"]             # [B,T,N,C]
        # 整段 rollout（去掉 t=0 的初值）
        p_all, t_all = out[:, 1:], tgt[:, 1:]
        tot["mse"] += mse_metric(p_all, t_all)
        tot["rne"] += rne_metric(p_all, t_all)
        tot["mse_last"] += mse_metric(out[:, -1], tgt[:, -1])
        tot["rne_last"] += rne_metric(out[:, -1], tgt[:, -1])
        tot["n"] += 1
    n = max(tot["n"], 1)
    return {
        "mse": tot["mse"] / n,
        "rne": tot["rne"] / n,
        "mse_last": tot["mse_last"] / n,
        "rne_last": tot["rne_last"] / n,
        "num_batches": tot["n"],
    }


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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt.get("config", {})
    model = build_model(args.model, config).to(device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    model.load_state_dict(state)

    keys = pick_keys(data_path, args.split)
    ds = DGPdeDataset(data_path, train_time_steps=args.train_time_steps,
                      trajectory_keys=keys)
    loader = create_dg_loader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, sampler=None)

    res = evaluate(model, loader, device)
    res.update({"model": args.model, "n_nodes": args.n_nodes,
                "ckpt": str(args.ckpt), "split": args.split})
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[eval] 已写入 {args.out}")


if __name__ == "__main__":
    main()
