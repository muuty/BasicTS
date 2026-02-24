"""
Temporal Neighbors Contrastive Loss

Key idea: Treat temporally adjacent timesteps as soft positives.
- Same timestep, different augmentation = strong positive (weight=1.0)
- Adjacent timesteps (|dt| <= k) = soft positive (weight decays with distance)
- Different sequences = negatives

Inspired by TS2Vec's hierarchical contrastive learning.

Memory-efficient version: processes each node independently.
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def temporal_neighbors_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    temporal_weight: float = 0.5,
    neighbor_range: int = 3,
) -> torch.Tensor:
    """
    Temporal Neighbors Contrastive Loss (Memory Efficient).

    Uses all timesteps for contrastive learning, treating temporally
    adjacent timesteps as soft positives.

    For memory efficiency, processes each sample's temporal dimension
    independently rather than computing global similarity matrix.

    Args:
        z1: [B, T, N, D] - first view representations (all timesteps)
        z2: [B, T, N, D] - second view representations (all timesteps)
        temperature: temperature for softmax scaling
        temporal_weight: weight for temporal neighbor positives (0~1)
        neighbor_range: how many adjacent timesteps to consider as positives

    Returns:
        loss: scalar tensor
    """
    B, T, N, D = z1.shape

    # Reshape to [B*N, T, D] - treat each node independently
    z1 = z1.permute(0, 2, 1, 3).reshape(B * N, T, D)
    z2 = z2.permute(0, 2, 1, 3).reshape(B * N, T, D)

    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Instance contrastive loss (same timestep, different augmentation)
    # Memory efficient: per-sample temporal contrastive
    instance_loss = instance_contrastive_loss_efficient(z1, z2, temperature)

    # Temporal contrastive loss (adjacent timesteps as soft positives)
    temporal_loss = hierarchical_temporal_loss(z1, z2, temperature, neighbor_range)

    # Combined loss
    total_loss = instance_loss + temporal_weight * temporal_loss

    return total_loss


def instance_contrastive_loss_efficient(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Memory-efficient instance contrastive loss.

    Instead of computing full (B*N*T) x (B*N*T) similarity matrix,
    compute per-sample temporal contrastive loss: for each (batch, node),
    compare timesteps within the same sample.

    Args:
        z1: [B*N, T, D] - normalized representations
        z2: [B*N, T, D] - normalized representations
    """
    BN, T, D = z1.shape
    device = z1.device

    # Concatenate z1 and z2 along time dimension: [B*N, 2T, D]
    z_combined = torch.cat([z1, z2], dim=1)  # [B*N, 2T, D]

    # Compute similarity within each sample: [B*N, 2T, 2T]
    sim = torch.bmm(z_combined, z_combined.transpose(1, 2)) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * T, device=device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(mask, float('-inf'))

    # Positive pairs: z1[t] <-> z2[t] (same timestep, different augmentation)
    # z1 is at indices [0, T), z2 is at indices [T, 2T)
    # For z1[t] (index t), positive is z2[t] (index T+t)
    # For z2[t] (index T+t), positive is z1[t] (index t)
    labels = torch.cat([
        torch.arange(T, 2 * T, device=device),  # z1 -> z2
        torch.arange(0, T, device=device),       # z2 -> z1
    ]).unsqueeze(0).expand(BN, -1)  # [B*N, 2T]

    # Compute cross-entropy loss
    # Reshape for cross_entropy: [B*N * 2T, 2T]
    sim_flat = sim.reshape(BN * 2 * T, 2 * T)
    labels_flat = labels.reshape(BN * 2 * T)

    loss = F.cross_entropy(sim_flat, labels_flat)
    return loss


def hierarchical_temporal_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    neighbor_range: int = 3,
) -> torch.Tensor:
    """
    Hierarchical temporal contrastive loss.

    For each timestep t, treat timesteps within neighbor_range as soft positives.
    Weight decreases with temporal distance.

    Args:
        z1: [B*N, T, D] - normalized representations
        z2: [B*N, T, D] - normalized representations
        temperature: temperature scaling
        neighbor_range: number of adjacent timesteps to consider
    """
    BN, T, D = z1.shape
    device = z1.device

    # Use mean of z1 and z2 for temporal consistency
    z = (z1 + z2) / 2  # [B*N, T, D]

    # Compute temporal similarity matrix for each sample
    # z: [B*N, T, D] -> sim: [B*N, T, T]
    sim = torch.bmm(z, z.transpose(1, 2)) / temperature

    # Create temporal distance weights
    # Adjacent timesteps have higher weight, decreasing with distance
    temporal_dist = torch.abs(
        torch.arange(T, device=device).unsqueeze(0) -
        torch.arange(T, device=device).unsqueeze(1)
    ).float()  # [T, T]

    # Weight matrix: 1 for neighbors, 0 for far timesteps
    # Exponential decay: exp(-dist / neighbor_range)
    weights = torch.exp(-temporal_dist / neighbor_range)
    weights = weights * (temporal_dist <= neighbor_range).float()
    weights = weights * (temporal_dist > 0).float()  # Exclude self

    # Normalize weights to sum to 1 for each row
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    weights = weights.unsqueeze(0).expand(BN, -1, -1)  # [B*N, T, T]

    # Soft cross-entropy with temporal neighbors as soft positives
    # For each timestep, maximize similarity with neighbors
    log_softmax = F.log_softmax(sim, dim=-1)  # [B*N, T, T]

    # Weighted sum of log probabilities for positive pairs
    loss = -torch.sum(weights * log_softmax) / (BN * T)

    return loss


def temporal_neighbors_contrastive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    z1: torch.Tensor = None,
    z2: torch.Tensor = None,
    temperature: float = 0.1,
    temporal_weight: float = 0.5,
    neighbor_range: int = 3,
    null_val: float = np.nan,
    **kwargs
) -> torch.Tensor:
    """
    Wrapper for basicts framework compatibility.

    Args:
        prediction: [B, T, N, C] - model predictions (ignored)
        target: [B, T, N, C] - ground truth (ignored)
        z1: [B, T, N, D] - first view representations
        z2: [B, T, N, D] - second view representations
        temperature: temperature for softmax
        temporal_weight: weight for temporal neighbor loss
        neighbor_range: adjacent timesteps to consider as positives
        null_val: (ignored)
    """
    if z1 is None or z2 is None:
        raise ValueError("z1 and z2 must be provided")

    return temporal_neighbors_loss(
        z1, z2, temperature, temporal_weight, neighbor_range
    )


def get_temporal_neighbors_loss(
    temperature: float = 0.1,
    temporal_weight: float = 0.5,
    neighbor_range: int = 3,
):
    """Factory function for temporal neighbors contrastive loss."""
    return functools.partial(
        temporal_neighbors_contrastive_loss,
        temperature=temperature,
        temporal_weight=temporal_weight,
        neighbor_range=neighbor_range,
    )
