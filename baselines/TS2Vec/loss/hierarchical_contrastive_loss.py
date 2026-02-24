"""
Hierarchical Contrastive Loss for TS2Vec

Combines instance-level and temporal-level contrastive losses
at multiple scales using max-pooling.
"""
import torch
import torch.nn.functional as F


def instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Instance-level contrastive loss.

    Contrasts different instances at the same timestamp.

    Args:
        z1, z2: Representations [B, T, D] or [B, T, N, D]

    Returns:
        Scalar loss
    """
    # Handle spatial-temporal data: flatten N into B
    if z1.dim() == 4:
        B, T, N, D = z1.shape
        z1 = z1.permute(0, 2, 1, 3).reshape(B * N, T, D)
        z2 = z2.permute(0, 2, 1, 3).reshape(B * N, T, D)

    B, T, D = z1.shape
    if B == 1:
        return z1.new_tensor(0.)

    # Concatenate views: [2B, T, D]
    z = torch.cat([z1, z2], dim=0)
    z = z.transpose(0, 1)  # [T, 2B, D]

    # Compute similarity: [T, 2B, 2B]
    sim = torch.matmul(z, z.transpose(1, 2))

    # Remove diagonal and compute log softmax
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    # Positive pairs: (i, B+i) and (B+i, i)
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

    return loss


def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Temporal-level contrastive loss.

    Contrasts different timestamps within the same instance.

    Args:
        z1, z2: Representations [B, T, D] or [B, T, N, D]

    Returns:
        Scalar loss
    """
    # Handle spatial-temporal data: flatten N into B
    if z1.dim() == 4:
        B, T, N, D = z1.shape
        z1 = z1.permute(0, 2, 1, 3).reshape(B * N, T, D)
        z2 = z2.permute(0, 2, 1, 3).reshape(B * N, T, D)

    B, T, D = z1.shape
    if T == 1:
        return z1.new_tensor(0.)

    # Concatenate views along time: [B, 2T, D]
    z = torch.cat([z1, z2], dim=1)

    # Compute similarity: [B, 2T, 2T]
    sim = torch.matmul(z, z.transpose(1, 2))

    # Remove diagonal and compute log softmax
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    # Positive pairs: (t, T+t) and (T+t, t)
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2

    return loss


def hierarchical_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float = 0.5,
    temporal_unit: int = 0
) -> torch.Tensor:
    """
    Hierarchical contrastive loss combining instance and temporal contrasts
    at multiple scales.

    Args:
        z1, z2: Representations [B, T, N, D]
        alpha: Weight for instance loss (1-alpha for temporal loss)
        temporal_unit: Minimum temporal unit for contrast

    Returns:
        Scalar loss
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0

    # Handle spatial-temporal data
    if z1.dim() == 4:
        B, T, N, D = z1.shape
        z1_flat = z1.permute(0, 2, 1, 3).reshape(B * N, T, D)
        z2_flat = z2.permute(0, 2, 1, 3).reshape(B * N, T, D)
    else:
        z1_flat = z1
        z2_flat = z2

    while z1_flat.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1_flat, z2_flat)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1_flat, z2_flat)
        d += 1

        # Max pooling to next scale
        z1_flat = F.max_pool1d(z1_flat.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2_flat = F.max_pool1d(z2_flat.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1_flat.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1_flat, z2_flat)
        d += 1

    return loss / d


class TS2VecLoss(torch.nn.Module):
    """
    TS2Vec Loss module for integration with basicts framework.
    """

    def __init__(self, alpha: float = 0.5, temporal_unit: int = 0):
        super().__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return hierarchical_contrastive_loss(
            z1, z2,
            alpha=self.alpha,
            temporal_unit=self.temporal_unit
        )


def get_ts2vec_loss(alpha: float = 0.5, temporal_unit: int = 0):
    """Factory function for TS2Vec loss."""
    loss_fn = TS2VecLoss(alpha=alpha, temporal_unit=temporal_unit)

    def loss_wrapper(prediction, target, z1=None, z2=None, **kwargs):
        if z1 is None or z2 is None:
            return F.l1_loss(prediction, target)
        return loss_fn(z1, z2)

    return loss_wrapper
