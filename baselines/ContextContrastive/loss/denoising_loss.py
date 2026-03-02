"""
Denoising Pre-training Loss.

Two-component loss:
  - L_recon: MAE on corrupted nodes (learn to denoise)
  - L_passthrough: MAE on clean nodes (don't over-denoise)

The passthrough loss is CRITICAL: without it, prior encoders degraded clean
performance by 1-3%. This explicitly prevents that failure mode.
"""

import functools
import torch
import torch.nn.functional as F
import numpy as np


def denoising_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recon_pred: torch.Tensor = None,
    recon_target: torch.Tensor = None,
    noise_mask: torch.Tensor = None,
    null_val: float = np.nan,
    passthrough_weight: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Denoising loss: reconstruct clean signal from noisy input.

    Args:
        prediction: [B, T, N, C] dummy (ignored)
        target: [B, T, N, C] dummy (ignored)
        recon_pred: [B, T, N, n_physical] reconstructed channels
        recon_target: [B, T, N, n_physical] clean channels
        noise_mask: [N] bool, True = corrupted node
        passthrough_weight: weight for clean node loss (default 1.0)

    Returns:
        total loss
    """
    corrupted = noise_mask  # [N]
    clean = ~noise_mask     # [N]

    # L_recon: denoise corrupted nodes
    if corrupted.sum() > 0:
        error_corrupt = (recon_pred[:, :, corrupted] - recon_target[:, :, corrupted]).abs()
        l_recon = error_corrupt.mean()
    else:
        l_recon = torch.tensor(0.0, device=recon_pred.device)

    # L_passthrough: preserve clean nodes
    if clean.sum() > 0:
        error_clean = (recon_pred[:, :, clean] - recon_target[:, :, clean]).abs()
        l_pass = error_clean.mean()
    else:
        l_pass = torch.tensor(0.0, device=recon_pred.device)

    return l_recon + passthrough_weight * l_pass


def get_denoising_loss(passthrough_weight: float = 1.0):
    """Factory function for denoising loss."""
    return functools.partial(
        denoising_reconstruction_loss,
        passthrough_weight=passthrough_weight,
    )


def denoising_with_r_supervision_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recon_pred: torch.Tensor = None,
    recon_target: torch.Tensor = None,
    noise_mask: torch.Tensor = None,
    reliability: torch.Tensor = None,
    null_val: float = np.nan,
    passthrough_weight: float = 1.0,
    r_weight: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """
    Denoising loss + direct reliability supervision.

    r supervision: BCE(r, 1-corrupt_mask). Corrupt nodes → r=0, clean → r=1.
    """
    base = denoising_reconstruction_loss(
        prediction, target, recon_pred, recon_target, noise_mask,
        passthrough_weight=passthrough_weight,
    )

    if reliability is not None and noise_mask is not None:
        r_target = (~noise_mask).float()                        # [N], clean=1
        # Reshape r_target to match reliability: could be [B, N] or [B, T, N, 1]
        r = reliability
        if r.dim() == 4:
            # V3 path: [B, T, N, 1] → average over T, squeeze last dim → [B, N]
            r = r.mean(dim=1).squeeze(-1)
        r_target = r_target.unsqueeze(0).expand_as(r)           # [B, N]
        r_loss = F.binary_cross_entropy(r, r_target)
        return base + r_weight * r_loss

    return base


def get_denoising_loss_with_r_supervision(passthrough_weight: float = 1.0, r_weight: float = 0.1):
    """Factory function for denoising loss with reliability supervision."""
    return functools.partial(
        denoising_with_r_supervision_loss,
        passthrough_weight=passthrough_weight,
        r_weight=r_weight,
    )
