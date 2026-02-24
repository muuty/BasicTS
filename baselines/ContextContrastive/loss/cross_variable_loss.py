"""
Cross-Variable Reconstruction Loss for Pre-training.

Computes MAE only on masked variable positions.
Compatible with basicts metric_forward pattern (inspect.signature matching).
"""
import functools

import torch
import numpy as np


def cross_variable_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recon_pred: torch.Tensor = None,
    recon_target: torch.Tensor = None,
    recon_mask: torch.Tensor = None,
    null_val: float = np.nan,
    **kwargs
) -> torch.Tensor:
    """
    Cross-variable reconstruction loss.

    Args:
        prediction: [B, T, N, C] - dummy prediction (ignored)
        target: [B, T, N, C] - dummy target (ignored)
        recon_pred: [B, T, N, 3] - reconstructed physical variables
        recon_target: [B, T, N, 3] - original physical variables
        recon_mask: [B, 1, 1, 3] - one-hot mask (1 = masked, compute loss here)
        null_val: ignored (for API compatibility)

    Returns:
        loss: scalar MAE on masked positions
    """
    mask = recon_mask.expand_as(recon_pred)
    error = (recon_pred - recon_target).abs() * mask
    return error.sum() / mask.sum().clamp(min=1)


def get_cross_variable_loss():
    """Factory function for cross-variable reconstruction loss."""
    return functools.partial(cross_variable_reconstruction_loss)
