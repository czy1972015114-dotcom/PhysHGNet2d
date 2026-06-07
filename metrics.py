"""
metrics.py — 统一的误差度量函数（DGNet / PhysHGNet 共用）。

train_dgnet.py 通过 `from metrics import mse as mse_metric, rne as rne_metric`
调用本文件。原仓库缺失此文件会导致 ImportError，这里补齐。

约定（与 train_*.py / 评测脚本保持一致）：
    mse(pred, target)  -> float   逐元素均方误差  mean((p-t)^2)
    rne(pred, target)  -> float   相对 L2 误差     ||p-t||_2 / ||t||_2
    mae(pred, target)  -> float   逐元素平均绝对误差
    rmse(pred, target) -> float   sqrt(mse)

所有函数都接受任意形状、可广播的张量（例如 [B,N,C] 或 [B,T,N,C]），
内部使用 no_grad，返回 Python float，便于直接累加打印。
"""

from __future__ import annotations

import torch


def _as_tensor(x) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x


@torch.no_grad()
def mse(pred, target) -> float:
    """逐元素均方误差 mean((pred - target)^2)。"""
    pred = _as_tensor(pred).float()
    target = _as_tensor(target).float()
    return torch.mean((pred - target) ** 2).item()


@torch.no_grad()
def rmse(pred, target) -> float:
    """均方根误差 sqrt(MSE)。"""
    return float(mse(pred, target) ** 0.5)


@torch.no_grad()
def mae(pred, target) -> float:
    """逐元素平均绝对误差 mean(|pred - target|)。"""
    pred = _as_tensor(pred).float()
    target = _as_tensor(target).float()
    return torch.mean((pred - target).abs()).item()


@torch.no_grad()
def rne(pred, target, eps: float = 1e-8) -> float:
    """
    相对 L2 误差（Relative Norm Error）：
        ||pred - target||_2 / max(||target||_2, eps)

    这是 PDE 算子学习里最常报告的相对误差指标，与 train_physhgnet.py
    内部的 rel_err 定义完全一致。
    """
    pred = _as_tensor(pred).float()
    target = _as_tensor(target).float()
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target).clamp(min=eps)
    return (num / den).item()


# 便于 `from metrics import *`
__all__ = ["mse", "rmse", "mae", "rne"]
