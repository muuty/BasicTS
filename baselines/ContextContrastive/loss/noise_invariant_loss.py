"""
Noise-Invariant Pre-training Loss.

Three components:
  - L_recon: MAE(encoder(x_noisy), x_clean) on corrupted nodes — denoising
  - L_pass:  MAE(encoder(x_noisy), x_clean) on clean nodes — passthrough
  - L_align: MSE(encoder(x_noisy), encoder(x_clean)) — noise invariance

Total: L_recon + passthrough_weight * L_pass + align_weight * L_align
"""

import functools
import torch
import numpy as np


def noise_invariant_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recon_noisy: torch.Tensor = None,
    recon_clean: torch.Tensor = None,
    recon_target: torch.Tensor = None,
    noise_mask: torch.Tensor = None,
    null_val: float = np.nan,
    passthrough_weight: float = 1.0,
    align_weight: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    corrupted = noise_mask
    clean = ~noise_mask

    # L_recon: denoise corrupted nodes
    if corrupted.sum() > 0:
        l_recon = (recon_noisy[:, :, corrupted] - recon_target[:, :, corrupted]).abs().mean()
    else:
        l_recon = torch.tensor(0.0, device=recon_noisy.device)

    # L_pass: preserve clean nodes
    if clean.sum() > 0:
        l_pass = (recon_noisy[:, :, clean] - recon_target[:, :, clean]).abs().mean()
    else:
        l_pass = torch.tensor(0.0, device=recon_noisy.device)

    # L_align: noise-invariant representation
    l_align = torch.nn.functional.mse_loss(recon_noisy, recon_clean)

    return l_recon + passthrough_weight * l_pass + align_weight * l_align


def get_noise_invariant_loss(passthrough_weight: float = 1.0, align_weight: float = 0.1):
    return functools.partial(
        noise_invariant_loss,
        passthrough_weight=passthrough_weight,
        align_weight=align_weight,
    )
