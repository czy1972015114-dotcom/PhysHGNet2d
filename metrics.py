"""
metrics.py — Unified evaluation metrics for PhysHGNet / DGNet.

Two metrics are used everywhere (training logs, ablation, scaling, speed,
baseline comparison) so numbers are directly comparable:

  MSE  : mean squared error              mean( (pred - target)^2 )
  RNE  : relative L2 (normalised) error  ||pred - target||_2 / ||target||_2

Both accept tensors of any matching shape. `which_step` lets callers score a
particular roll-out step (e.g. the final predicted field) or the whole
trajectory. All functions are torch-only and grad-free.
"""
from typing import Dict
import torch


@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


@torch.no_grad()
def rne(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Relative (normalised) L2 error over the flattened tensors."""
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target).clamp(min=eps)
    return (num / den).item()


@torch.no_grad()
def compute_all(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Return {'mse':.., 'rne':..} for a (B,T,N,C) / (B,N,C) / (N,) tensor pair."""
    return {"mse": mse(pred, target), "rne": rne(pred, target)}


@torch.no_grad()
def trajectory_metrics(u_pred: torch.Tensor, u_true: torch.Tensor) -> Dict[str, float]:
    """
    Metrics for a full roll-out. Expects (B, T, N, C).
    Reports both the final-step error (the hardest, long-horizon target) and the
    mean-over-all-steps error.
    """
    assert u_pred.shape == u_true.shape, f"{u_pred.shape} vs {u_true.shape}"
    out = {}
    out["mse_final"] = mse(u_pred[:, -1], u_true[:, -1])
    out["rne_final"] = rne(u_pred[:, -1], u_true[:, -1])
    out["mse_all"] = mse(u_pred, u_true)
    out["rne_all"] = rne(u_pred, u_true)
    return out
