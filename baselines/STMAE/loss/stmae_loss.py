"""
Loss functions for STMAE pre-training.

StructureLoss: BCE loss for structure/adjacency reconstruction
FeatureLoss: MAE loss for feature reconstruction
STMAELoss: Combined loss for pre-training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from typing import Dict, Optional


class StructureLoss(nn.Module):
    """
    Structure reconstruction loss using Binary Cross Entropy.

    Computes BCE loss on masked edge positions, where the target is
    to predict connectivity (all ones).
    """

    def __init__(self, loss_type: str = 'cls_boost'):
        """
        Args:
            loss_type: Type of loss computation
                - 'cls_boost': BCE on masked positions only
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        s_mask: torch.Tensor,
        null_val: float = None,
    ) -> torch.Tensor:
        """
        Compute structure reconstruction loss.

        Args:
            prediction: Predicted adjacency logits [B, N, N]
            target: Target support matrix [N, N] or [B, N, N]
            s_mask: Structure mask [N, N] where 1=keep, 0=masked
            null_val: Not used (for interface compatibility)

        Returns:
            loss: Scalar loss value
        """
        B = prediction.shape[0]

        if self.loss_type == 'cls_boost':
            # Target: all ones (predict full connectivity)
            if target.dim() == 2:
                target = torch.ones_like(target)
                target = repeat(target, 'm n -> b m n', b=B)
            else:
                target = torch.ones_like(target)

            # Inverse mask: 1 where masked (positions we want to reconstruct)
            if s_mask.dim() == 2:
                inv_mask = repeat(1 - s_mask, 'm n -> b m n', b=B)
            else:
                inv_mask = 1 - s_mask

            # Compute BCE loss on masked positions
            if torch.sum(inv_mask) > 0:
                diff = F.binary_cross_entropy_with_logits(
                    prediction, target, reduction='none'
                ) * inv_mask
                loss = torch.sum(diff) / torch.sum(inv_mask)
            else:
                # No masking - compute on all positions
                diff = F.binary_cross_entropy_with_logits(
                    prediction, target, reduction='none'
                )
                loss = torch.mean(diff)
        else:
            raise ValueError(f"Unknown structure loss type: {self.loss_type}")

        return loss


class FeatureLoss(nn.Module):
    """
    Feature reconstruction loss using Mean Absolute Error (L1).

    Computes MAE on masked temporal positions.
    """

    def __init__(self, loss_type: str = 'reg_l1'):
        """
        Args:
            loss_type: Type of loss computation
                - 'reg_l1': L1/MAE loss on masked positions
                - 'reg_l2': L2/MSE loss on masked positions
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        f_mask: torch.Tensor,
        null_val: float = None,
    ) -> torch.Tensor:
        """
        Compute feature reconstruction loss.

        Args:
            prediction: Predicted features [B, T, N, D] or [B, T, N]
            target: Target features [B, T, N, D] or [B, T, N]
            f_mask: Feature mask [B, T, N] where 1=keep, 0=masked
            null_val: Value to treat as null (masked in loss computation)

        Returns:
            loss: Scalar loss value
        """
        # Handle multi-channel: use only first channel
        if prediction.dim() == 4 and prediction.shape[-1] > 1:
            prediction = prediction[..., :1]
        if target.dim() == 4 and target.shape[-1] > 1:
            target = target[..., :1]

        # Squeeze last dimension if present
        if prediction.dim() == 4:
            prediction = prediction.squeeze(-1)
        if target.dim() == 4:
            target = target.squeeze(-1)

        # Inverse mask: 1 where masked (positions we want to reconstruct)
        inv_mask = 1 - f_mask

        # Handle null values if specified
        if null_val is not None:
            if torch.isnan(torch.tensor(null_val)):
                valid_mask = ~torch.isnan(target)
            else:
                valid_mask = ~torch.isclose(
                    target, torch.tensor(null_val).to(target.device),
                    atol=1e-5
                )
            inv_mask = inv_mask * valid_mask.float()

        if self.loss_type == 'reg_l1':
            if torch.sum(inv_mask) > 0:
                diff = torch.abs(prediction - target) * inv_mask
                loss = torch.sum(diff) / torch.sum(inv_mask)
            else:
                diff = torch.abs(prediction - target)
                loss = torch.mean(diff)

        elif self.loss_type == 'reg_l2':
            if torch.sum(inv_mask) > 0:
                diff = (prediction - target) ** 2 * inv_mask
                loss = torch.sum(diff) / torch.sum(inv_mask)
            else:
                diff = (prediction - target) ** 2
                loss = torch.mean(diff)
        else:
            raise ValueError(f"Unknown feature loss type: {self.loss_type}")

        return loss


class STMAELoss(nn.Module):
    """
    Combined STMAE loss for pre-training.

    Combines structure and feature reconstruction losses with configurable weights.
    """

    def __init__(
        self,
        sl_weight: float = 1.0,
        fl_weight: float = 1.0,
        sl_type: str = 'cls_boost',
        fl_type: str = 'reg_l1',
    ):
        """
        Args:
            sl_weight: Weight for structure loss
            fl_weight: Weight for feature loss
            sl_type: Type of structure loss
            fl_type: Type of feature loss
        """
        super().__init__()
        self.sl_weight = sl_weight
        self.fl_weight = fl_weight

        self.structure_loss = StructureLoss(loss_type=sl_type)
        self.feature_loss = FeatureLoss(loss_type=fl_type)

    def forward(
        self,
        prediction: Dict,
        target: torch.Tensor = None,
        null_val: float = None,
        support: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute combined STMAE loss.

        Args:
            prediction: Dictionary containing model outputs:
                - 'reconstruction': Reconstructed features [B, T, N, D]
                - 's_mask': Structure mask [N, N]
                - 'f_mask': Feature mask [B, T, N]
                - 'adj_pred': Predicted adjacency [B, N, N] (optional)
            target: Target features [B, T, N, D]
            null_val: Null value for masking
            support: Target adjacency matrix [N, N]

        Returns:
            loss: Combined loss value
        """
        # Extract from prediction dict if needed
        if isinstance(prediction, dict):
            recon_f = prediction.get('reconstruction')
            recon_s = prediction.get('adj_pred', prediction.get('recon_s'))
            f_mask = prediction.get('f_mask')
            s_mask = prediction.get('s_mask')

            # If target not provided, try to get from prediction
            if target is None:
                target = prediction.get('target')
            if support is None:
                support = prediction.get('support')
        else:
            raise ValueError("prediction must be a dictionary")

        loss = torch.tensor(0.0, device=recon_f.device)
        loss_info = {}

        # Feature loss
        if recon_f is not None and target is not None and f_mask is not None:
            f_loss = self.feature_loss(recon_f, target, f_mask, null_val)
            loss = loss + self.fl_weight * f_loss
            loss_info['f_loss'] = f_loss.item()

        # Structure loss
        if recon_s is not None and support is not None and s_mask is not None:
            s_loss = self.structure_loss(recon_s, support, s_mask, null_val)
            loss = loss + self.sl_weight * s_loss
            loss_info['s_loss'] = s_loss.item()

        return loss


def stmae_pretrain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
    **kwargs
) -> torch.Tensor:
    """
    Simple loss function for STMAE pre-training.

    This is a convenience function that extracts the 'loss' key from the model output.
    The actual loss computation is done inside the STMAE model.

    Args:
        prediction: Model output dictionary with 'loss' key
        target: Target tensor (not used, loss computed inside model)
        null_val: Null value (not used)

    Returns:
        loss: Pre-computed loss from model
    """
    if isinstance(prediction, dict) and 'loss' in prediction:
        return prediction['loss']
    raise ValueError("prediction must be a dict with 'loss' key for pretrain loss")


def stmae_finetune_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
) -> torch.Tensor:
    """
    MAE loss function for STMAE fine-tuning.

    Standard masked MAE loss for forecasting.

    Args:
        prediction: Predicted values [B, T, N, D]
        target: Target values [B, T, N, D]
        null_val: Value to treat as null/missing

    Returns:
        loss: MAE loss value
    """
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(target)
    else:
        eps = 1e-5
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val).expand_as(target).to(target.device),
            atol=eps
        )

    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(prediction - target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)
