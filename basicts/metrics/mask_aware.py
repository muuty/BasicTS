import torch


def mask_aware_mae(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """MAE computed only on valid (non-missing) targets using explicit mask.

    Unlike masked_mae(null_val=0) which excludes ALL target=0 (both missing AND real zero),
    this uses the explicit mask channel to exclude ONLY truly missing targets.

    Args:
        prediction: (B, L, N, C) model predictions
        target: (B, L, N, C) target values (0 for missing)
        target_mask: (B, L, N, 1) binary mask, 1=valid, 0=missing
    """

    mask = (target_mask == 1).float()
    ae = torch.abs(prediction - target)
    return (ae * mask).sum() / mask.sum().clamp(min=1)


def mask_aware_mape(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """MAPE computed only on valid (non-missing) targets using explicit mask.

    Also excludes target=0 to avoid division by zero (same as standard MAPE).
    """

    mask = (target_mask == 1).float() * (target != 0).float()
    ape = torch.abs((prediction - target) / target.clamp(min=1e-8))
    return (ape * mask).sum() / mask.sum().clamp(min=1)


def mask_aware_rmse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """RMSE computed only on valid (non-missing) targets using explicit mask."""

    mask = (target_mask == 1).float()
    se = (prediction - target) ** 2
    mse = (se * mask).sum() / mask.sum().clamp(min=1)
    return torch.sqrt(mse)
