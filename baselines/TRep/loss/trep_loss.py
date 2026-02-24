"""
T-Rep Loss: Combines contrastive loss with temporal relation prediction tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.TS2Vec.loss import hierarchical_contrastive_loss


class TRepLoss(nn.Module):
    """
    T-Rep Loss combining:
    1. Instance contrastive loss
    2. Temporal contrastive loss
    3. Temporal relation prediction (JSD)
    4. Conditional representation prediction
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temporal_unit: int = 0,
        instance_weight: float = 0.25,
        temporal_weight: float = 0.25,
        jsd_weight: float = 0.25,
        cond_weight: float = 0.25,
    ):
        super().__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.instance_weight = instance_weight
        self.temporal_weight = temporal_weight
        self.jsd_weight = jsd_weight
        self.cond_weight = cond_weight

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        tau1: torch.Tensor = None,
        tau2: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute T-Rep loss.

        Args:
            z1, z2: Representations [B, T, N, D]
            tau1, tau2: Time embeddings [B, T, N, time_embed_dim]
        """
        # Hierarchical contrastive loss (instance + temporal)
        contrastive_loss = hierarchical_contrastive_loss(
            z1, z2,
            alpha=self.alpha,
            temporal_unit=self.temporal_unit
        )

        # If no time embeddings, return only contrastive loss
        if tau1 is None or tau2 is None:
            return contrastive_loss

        # JSD-based temporal relation loss
        # Measures consistency of temporal embeddings between views
        jsd_loss = self._jsd_loss(tau1, tau2)

        # Conditional prediction loss
        # Predict representation from time embedding
        cond_loss = self._conditional_loss(z1, tau1, z2, tau2)

        # Combine losses
        total_loss = (
            (self.instance_weight + self.temporal_weight) * contrastive_loss +
            self.jsd_weight * jsd_loss +
            self.cond_weight * cond_loss
        )

        return total_loss

    def _jsd_loss(self, tau1: torch.Tensor, tau2: torch.Tensor) -> torch.Tensor:
        """Jensen-Shannon Divergence based loss for time embeddings."""
        # Normalize to probability distributions
        p1 = F.softmax(tau1, dim=-1)
        p2 = F.softmax(tau2, dim=-1)

        # JSD = 0.5 * KL(p1 || m) + 0.5 * KL(p2 || m) where m = 0.5 * (p1 + p2)
        m = 0.5 * (p1 + p2)
        jsd = 0.5 * (F.kl_div(m.log(), p1, reduction='batchmean') +
                     F.kl_div(m.log(), p2, reduction='batchmean'))

        return jsd

    def _conditional_loss(
        self,
        z1: torch.Tensor,
        tau1: torch.Tensor,
        z2: torch.Tensor,
        tau2: torch.Tensor
    ) -> torch.Tensor:
        """
        Conditional prediction loss.

        The idea is that representations at the same time should be similar
        regardless of which view they came from.
        """
        # L2 distance between representations at corresponding positions
        loss = F.mse_loss(z1, z2)
        return loss


def get_trep_loss(
    alpha: float = 0.5,
    temporal_unit: int = 0,
    instance_weight: float = 0.25,
    temporal_weight: float = 0.25,
    jsd_weight: float = 0.25,
    cond_weight: float = 0.25,
):
    """Factory function for T-Rep loss."""
    loss_fn = TRepLoss(
        alpha=alpha,
        temporal_unit=temporal_unit,
        instance_weight=instance_weight,
        temporal_weight=temporal_weight,
        jsd_weight=jsd_weight,
        cond_weight=cond_weight,
    )

    def loss_wrapper(prediction, target, z1=None, z2=None, tau1=None, tau2=None, **kwargs):
        if z1 is None or z2 is None:
            return F.l1_loss(prediction, target)
        return loss_fn(z1, z2, tau1, tau2)

    return loss_wrapper
