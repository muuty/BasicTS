"""
GPT-ST Loss Functions.

Includes:
- Reconstruction loss (masked MAE)
- Classification loss for adaptive masking
- Combined pre-training loss
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    null_val: float = 0.0,
) -> torch.Tensor:
    """Compute MAE loss only on masked (reconstructed) positions.

    Args:
        prediction: Predicted values [B, T, N, D]
        target: Ground truth values [B, T, N, D]
        mask: Binary mask where 1 indicates masked positions [B, T, N, D]
        null_val: Value to treat as null/missing

    Returns:
        Scalar loss value
    """
    # Ensure mask has same shape as prediction
    if mask.dim() < prediction.dim():
        mask = mask.unsqueeze(-1).expand_as(prediction)

    # Compute absolute error
    abs_error = torch.abs(prediction - target)

    # Apply mask (only compute loss on masked positions)
    masked_error = abs_error * mask

    # Handle null values
    if null_val is not None and not torch.isnan(torch.tensor(null_val)):
        valid_mask = (target != null_val).float() * mask
    else:
        valid_mask = (~torch.isnan(target)).float() * mask

    # Compute mean over valid masked positions
    loss = masked_error.sum() / (valid_mask.sum() + 1e-8)

    return loss


def classification_loss(
    probability: torch.Tensor,
    spatial_heads: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute classification loss for adaptive masking.

    This loss encourages the MLP_RL classifier to predict which
    spatial hyperedge head each node belongs to.

    Args:
        probability: Classification probabilities from MLP_RL [B, T, N, HS]
        spatial_heads: Spatial hyperedge assignments (already transposed) [B, T, N, HS]
        temperature: Temperature for softmax

    Returns:
        Scalar loss value
    """
    # spatial_heads is already [B, T, N, HS] from model (transposed in gptst_arch.py line 235)
    target_distribution = spatial_heads

    # Normalize to get target probabilities
    target_distribution = F.softmax(target_distribution / temperature, dim=-1)

    # KL divergence loss
    log_prob = F.log_softmax(probability, dim=-1)
    kl_loss = F.kl_div(log_prob, target_distribution, reduction='batchmean')

    return kl_loss


class GPTSTPretrainLoss(nn.Module):
    """Combined pre-training loss for GPT-ST.

    Combines:
    1. Reconstruction loss (masked MAE)
    2. Classification loss (optional, for adaptive masking)
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        class_weight: float = 0.1,
        use_classification_loss: bool = True,
        null_val: float = 0.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            recon_weight: Weight for reconstruction loss
            class_weight: Weight for classification loss
            use_classification_loss: Whether to use classification loss
            null_val: Null value in data
            temperature: Temperature for classification softmax
        """
        super(GPTSTPretrainLoss, self).__init__()

        self.recon_weight = recon_weight
        self.class_weight = class_weight
        self.use_classification_loss = use_classification_loss
        self.null_val = null_val
        self.temperature = temperature

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        probability: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute combined pre-training loss.

        Args:
            prediction: Reconstructed values [B, T, N, D]
            target: Ground truth values [B, T, N, D]
            mask: Binary mask (1 = masked) [B, T, N, D]
            probability: Classification probabilities [B, T, N, HS]
            hidden_states: Spatial head assignments [B, T, HS, N]

        Returns:
            Total loss
        """
        total_loss = 0.0

        # Reconstruction loss
        if mask is not None:
            recon_loss = masked_mae_loss(prediction, target, mask, self.null_val)
        else:
            # Fallback to standard MAE
            recon_loss = F.l1_loss(prediction, target)

        total_loss = total_loss + self.recon_weight * recon_loss

        # Classification loss
        if self.use_classification_loss and probability is not None and hidden_states is not None:
            class_loss = classification_loss(probability, hidden_states, self.temperature)
            total_loss = total_loss + self.class_weight * class_loss

        return total_loss


def gptst_pretrain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    probability: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    null_val: float = 0.0,
    recon_weight: float = 1.0,
    class_weight: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Functional interface for GPT-ST pre-training loss.

    Args:
        prediction: Reconstructed values [B, T, N, D]
        target: Ground truth values [B, T, N, D]
        mask: Binary mask (1 = masked) [B, T, N, D]
        probability: Classification probabilities [B, T, N, HS]
        hidden_states: Spatial head assignments [B, T, HS, N]
        null_val: Null value in data
        recon_weight: Weight for reconstruction loss
        class_weight: Weight for classification loss

    Returns:
        Total loss
    """
    total_loss = 0.0

    # Reconstruction loss
    if mask is not None:
        recon_loss = masked_mae_loss(prediction, target, mask, null_val)
    else:
        recon_loss = F.l1_loss(prediction, target)

    total_loss = total_loss + recon_weight * recon_loss

    # Classification loss
    if probability is not None and hidden_states is not None:
        class_loss = classification_loss(probability, hidden_states)
        total_loss = total_loss + class_weight * class_loss

    return total_loss


def gptst_finetune_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Loss function for GPT-ST fine-tuning (standard MAE).

    Args:
        prediction: Predicted values [B, T, N, D]
        target: Ground truth values [B, T, N, D]
        null_val: Null value in data

    Returns:
        MAE loss
    """
    if null_val is not None and not torch.isnan(torch.tensor(null_val)):
        mask = target != null_val
    else:
        mask = ~torch.isnan(target)

    mask = mask.float()
    abs_error = torch.abs(prediction - target) * mask
    loss = abs_error.sum() / (mask.sum() + 1e-8)

    return loss
