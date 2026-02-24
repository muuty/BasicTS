"""
Contrastive Loss for Stage 1 Pre-training.

SimCLR-style InfoNCE loss:
- Positive pair: same sample with different augmentations (z1[i], z2[i])
- Negative pairs: all other samples in the batch
"""
import functools

import torch
import torch.nn.functional as F
import numpy as np


def simclr_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    SimCLR-style contrastive loss (NT-Xent).

    Args:
        z1: [B, N, D] - first view representations
        z2: [B, N, D] - second view representations
        temperature: temperature for softmax scaling

    Returns:
        loss: scalar tensor
    """
    B, N, D = z1.shape

    # Flatten: [B*N, D]
    z1_flat = F.normalize(z1.reshape(B * N, D), dim=-1)
    z2_flat = F.normalize(z2.reshape(B * N, D), dim=-1)

    # Total samples
    n_samples = B * N

    # Concatenate both views: [2*B*N, D]
    z = torch.cat([z1_flat, z2_flat], dim=0)

    # Similarity matrix: [2*B*N, 2*B*N]
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * n_samples, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))

    # Positive pairs: (z1[i], z2[i]) and (z2[i], z1[i])
    # For z1[i], positive is z2[i] at index n_samples + i
    # For z2[i], positive is z1[i] at index i
    labels = torch.cat([
        torch.arange(n_samples, 2 * n_samples, device=z.device),  # z1 -> z2
        torch.arange(0, n_samples, device=z.device),              # z2 -> z1
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


def contrastive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    z1: torch.Tensor = None,
    z2: torch.Tensor = None,
    temperature: float = 0.1,
    null_val: float = np.nan,
    **kwargs
) -> torch.Tensor:
    """
    Contrastive loss function compatible with basicts framework.

    Args:
        prediction: [B, T, N, C] - model predictions (ignored for contrastive)
        target: [B, T, N, C] - ground truth (ignored for contrastive)
        z1: [B, N, D] - first view representations
        z2: [B, N, D] - second view representations
        temperature: temperature for softmax scaling
        null_val: (ignored, for API compatibility)

    Returns:
        loss: scalar tensor
    """
    if z1 is None or z2 is None:
        raise ValueError("z1 and z2 must be provided for contrastive loss")

    return simclr_loss(z1, z2, temperature)


def get_contrastive_loss(temperature: float = 0.1):
    """Factory function for contrastive loss."""
    return functools.partial(
        contrastive_loss,
        temperature=temperature
    )
