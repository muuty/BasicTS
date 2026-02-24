"""
Denoising Pre-training Loss with Learned Reliability Estimation.

Unified loss that jointly trains denoising and reliability:
  L = mean(r * ||denoised - clean||^2 + beta * (-log(r + eps)))
    + passthrough_weight * L_passthrough

- Clean nodes: small error -> r stays near 1 (beta regularizer dominates)
- Noisy with good denoising: small error -> r stays high
- Noisy with failed denoising (stuck etc): large error -> r drops -> "don't trust"
- beta prevents trivial r=0 solution

Optimal r* for given error e: r* = beta / (e^2 + beta)
  => Smooth monotonic mapping from reconstruction error to reliability.

Theoretical connection: heteroscedastic aleatoric uncertainty (Kendall & Gal 2018).
"""

import functools
import torch
import numpy as np


def denoising_reliability_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recon_pred: torch.Tensor = None,
    recon_target: torch.Tensor = None,
    noise_mask: torch.Tensor = None,
    reliability: torch.Tensor = None,
    null_val: float = np.nan,
    passthrough_weight: float = 1.0,
    reliability_beta: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Reliability-weighted denoising loss.

    Args:
        recon_pred: [B, T, N, n_physical] reconstructed channels
        recon_target: [B, T, N, n_physical] clean channels
        noise_mask: [N] bool, True = corrupted node
        reliability: [B, T, N, 1] reliability scores from encoder (0-1)
        passthrough_weight: weight for clean node passthrough loss
        reliability_beta: entropy regularizer weight (controls r* curve)
    """
    eps = 1e-6
    r = reliability.squeeze(-1)  # [B, T, N]

    # Per-node, per-timestep squared error (averaged across physical channels)
    error_sq = ((recon_pred - recon_target) ** 2).mean(dim=-1)  # [B, T, N]

    # Unified reliability-weighted loss (handles both noisy and clean nodes)
    l_reliability = (r * error_sq + reliability_beta * (-torch.log(r + eps))).mean()

    # L_passthrough: preserve clean node fidelity (MAE)
    clean = ~noise_mask  # [N]
    if clean.sum() > 0:
        error_clean = (recon_pred[:, :, clean] - recon_target[:, :, clean]).abs()
        l_pass = error_clean.mean()
    else:
        l_pass = torch.tensor(0.0, device=recon_pred.device)

    return l_reliability + passthrough_weight * l_pass


def get_denoising_reliability_loss(passthrough_weight: float = 1.0, reliability_beta: float = 1.0):
    """Factory function for reliability-weighted denoising loss."""
    return functools.partial(
        denoising_reliability_loss,
        passthrough_weight=passthrough_weight,
        reliability_beta=reliability_beta,
    )
