"""
Temporal-Aware Contrastive Loss

Improves upon standard InfoNCE by using temporal context (tod, dow)
for smarter negative sampling:
- Same tod/dow samples are treated as soft positives (not pushed away as hard)
- Different tod/dow samples are treated as hard negatives

This encourages the model to learn representations that distinguish
different temporal patterns (e.g., rush hour vs. midnight).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAwareContrastiveLoss(nn.Module):
    """
    Temporal-Aware Contrastive Loss with context-weighted negative sampling.

    Key idea: Samples with similar temporal context (tod, dow) should not
    be pushed apart as strongly as samples with different contexts.

    Args:
        temperature: Temperature for softmax scaling
        tod_weight: Weight for time-of-day difference in negative weighting
        dow_weight: Weight for day-of-week difference in negative weighting
        soft_positive_weight: Weight applied to same-context negatives (0-1)
                             0 = treat as positive, 1 = treat as hard negative
    """

    def __init__(
        self,
        temperature: float = 0.1,
        tod_weight: float = 0.5,
        dow_weight: float = 0.5,
        soft_positive_weight: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.tod_weight = tod_weight
        self.dow_weight = dow_weight
        self.soft_positive_weight = soft_positive_weight

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        tod: torch.Tensor = None,
        dow: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute temporal-aware contrastive loss.

        Args:
            z1: First view embeddings [B, N, D] or [B, D]
            z2: Second view embeddings [B, N, D] or [B, D]
            tod: Time of day indices [B] or [B, T] (uses last timestep)
            dow: Day of week indices [B] or [B, T] (uses last timestep)

        Returns:
            Scalar loss value
        """
        # Handle different input shapes
        if z1.dim() == 3:
            # [B, N, D] -> [B*N, D]
            B, N, D = z1.shape
            z1 = z1.reshape(B * N, D)
            z2 = z2.reshape(B * N, D)

            # Expand tod/dow for each node
            if tod is not None:
                if tod.dim() == 2:
                    tod = tod[:, -1]  # Use last timestep
                tod = tod.unsqueeze(1).expand(B, N).reshape(B * N)
            if dow is not None:
                if dow.dim() == 2:
                    dow = dow[:, -1]
                dow = dow.unsqueeze(1).expand(B, N).reshape(B * N)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.shape[0]

        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature  # [B, B]

        # Create negative weights based on temporal context
        if tod is not None or dow is not None:
            negative_weights = self._compute_negative_weights(tod, dow, batch_size, z1.device)
        else:
            # Fall back to standard InfoNCE if no temporal info
            negative_weights = torch.ones(batch_size, batch_size, device=z1.device)

        # Mask out diagonal (positive pairs)
        mask = torch.eye(batch_size, device=z1.device).bool()
        negative_weights = negative_weights.masked_fill(mask, 0)

        # Positive similarities (diagonal)
        pos_sim = torch.diag(sim_matrix)

        # Weighted negative similarities
        # Apply weights to exp(sim) before summing
        exp_sim = torch.exp(sim_matrix)
        weighted_neg_sum = (exp_sim * negative_weights).sum(dim=1)

        # InfoNCE loss with weighted negatives
        loss = -pos_sim + torch.log(torch.exp(pos_sim) + weighted_neg_sum + 1e-8)

        return loss.mean()

    def _compute_negative_weights(
        self,
        tod: torch.Tensor,
        dow: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute negative weights based on temporal context similarity.

        Same tod/dow -> lower weight (soft positive)
        Different tod/dow -> higher weight (hard negative)
        """
        weights = torch.ones(batch_size, batch_size, device=device)

        if tod is not None:
            # tod difference: same hour -> 0, different hour -> 1
            # Discretize to hours (assuming 288 steps per day, 12 per hour)
            tod_hours = (tod.float() / 12).long()
            tod_same = (tod_hours.unsqueeze(0) == tod_hours.unsqueeze(1)).float()
            tod_diff = 1 - tod_same

            # Weight: soft_positive for same tod, 1.0 for different tod
            tod_weight = tod_same * self.soft_positive_weight + tod_diff * 1.0
            weights = weights * (self.tod_weight * tod_weight + (1 - self.tod_weight))

        if dow is not None:
            # dow difference: same day -> 0, different day -> 1
            dow_same = (dow.unsqueeze(0) == dow.unsqueeze(1)).float()
            dow_diff = 1 - dow_same

            # Weekday vs weekend distinction
            # 0-4: weekday, 5-6: weekend
            is_weekday = (dow < 5).float()
            same_type = (is_weekday.unsqueeze(0) == is_weekday.unsqueeze(1)).float()

            # Weight: soft_positive for same dow, higher for different dow
            # Extra penalty for weekday vs weekend
            dow_weight = dow_same * self.soft_positive_weight + dow_diff * same_type + dow_diff * (1 - same_type) * 1.2
            weights = weights * (self.dow_weight * dow_weight + (1 - self.dow_weight))

        return weights


def get_temporal_aware_contrastive_loss(
    temperature: float = 0.1,
    tod_weight: float = 0.5,
    dow_weight: float = 0.5,
    soft_positive_weight: float = 0.3,
):
    """Factory function for temporal-aware contrastive loss."""
    loss_fn = TemporalAwareContrastiveLoss(
        temperature=temperature,
        tod_weight=tod_weight,
        dow_weight=dow_weight,
        soft_positive_weight=soft_positive_weight,
    )

    def loss_wrapper(prediction, target, z1=None, z2=None, tod=None, dow=None, **kwargs):
        if z1 is None or z2 is None:
            # Fallback to MAE if no contrastive embeddings
            return F.l1_loss(prediction, target)
        return loss_fn(z1, z2, tod=tod, dow=dow)

    return loss_wrapper
