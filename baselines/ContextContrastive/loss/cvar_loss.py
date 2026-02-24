import numpy as np
import torch


def masked_cvar_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, alpha: float = 0.1) -> torch.Tensor:
    """CVaR (Conditional Value at Risk) MAE Loss.

    Only backpropagates through the worst alpha fraction of samples.
    For alpha=0.1, this optimizes the worst 10% of per-sample MAE.

    Args:
        prediction: [B, T, N, 1]
        target: [B, T, N, 1]
        null_val: value to mask out
        alpha: fraction of worst samples to optimize (default 0.1 = worst 10%)
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()

    # Per-sample MAE: average over T, N, C dimensions
    abs_error = torch.abs(prediction - target)
    abs_error = abs_error * mask
    abs_error = torch.nan_to_num(abs_error)

    # [B] - mean error per sample
    valid_counts = mask.sum(dim=(1, 2, 3)).clamp(min=1)
    per_sample_mae = abs_error.sum(dim=(1, 2, 3)) / valid_counts

    # Take worst alpha fraction
    k = max(1, int(per_sample_mae.shape[0] * alpha))
    worst_k, _ = torch.topk(per_sample_mae, k)

    return worst_k.mean()


def masked_cvar_mae_20(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """CVaR MAE with alpha=0.2 (worst 20%)."""
    return masked_cvar_mae(prediction, target, null_val, alpha=0.2)
