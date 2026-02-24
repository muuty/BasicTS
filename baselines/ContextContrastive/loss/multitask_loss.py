"""Combined forecasting + cross-variable reconstruction loss for multi-task learning."""
import functools
import numpy as np
import torch


def multitask_crossvar_loss(prediction, target, recon_pred=None, recon_target=None,
                            recon_mask=None, lambda_recon=0.1, null_val=np.nan, **kwargs):
    """Combined loss: forecasting MAE + lambda * cross-variable reconstruction MAE.

    Args:
        prediction: forecasting output [B, T, N, 1]
        target: forecasting target [B, T, N, 1]
        recon_pred: reconstruction output [B, T, N, 3]
        recon_target: original physical variables [B, T, N, 3]
        recon_mask: one-hot mask of which variable was masked [B, 1, 1, 3]
        lambda_recon: weight for reconstruction loss
        null_val: value to ignore in forecasting loss
    """
    # Forecasting loss (masked MAE, exclude null_val)
    if np.isnan(null_val):
        forecast_mask = ~torch.isnan(target)
    else:
        forecast_mask = target != null_val
    forecast_error = (prediction - target).abs()
    forecast_loss = forecast_error[forecast_mask].mean()

    # Reconstruction loss (only on masked variable positions)
    mask = recon_mask.expand_as(recon_pred)
    recon_error = (recon_pred - recon_target).abs() * mask
    recon_loss = recon_error.sum() / mask.sum().clamp(min=1)

    return forecast_loss + lambda_recon * recon_loss


def get_multitask_crossvar_loss(lambda_recon=0.1):
    return functools.partial(multitask_crossvar_loss, lambda_recon=lambda_recon)
