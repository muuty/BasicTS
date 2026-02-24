"""
ST-SSL Loss Functions.

Combined loss for ST-SSL training:
- Prediction loss (MAE)
- Temporal contrastive loss
- Spatial contrastive loss
"""
import torch
import torch.nn as nn
from basicts.metrics import masked_mae


def stssl_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        repr1: torch.Tensor = None,
        repr2: torch.Tensor = None,
        model: nn.Module = None,
        null_val: float = 0.0,
        loss_weights: list = None,
        mode: str = 'end2end',
        **kwargs
) -> torch.Tensor:
    """
    Combined loss for ST-SSL.

    Args:
        prediction: predicted values, [B, L, N, C]
        target: ground truth values, [B, L, N, C]
        repr1: representation from view 1, [B, 1, N, D]
        repr2: representation from view 2, [B, 1, N, D]
        model: STSSL model instance for computing SSL losses
        null_val: null value for masking
        loss_weights: weights for [pred_loss, temporal_loss, spatial_loss]
        mode: training mode ('pretrain', 'finetune', 'end2end')

    Returns:
        total_loss: combined loss value
    """
    if loss_weights is None:
        loss_weights = [1.0, 1.0, 1.0]

    # Prediction loss
    pred_loss = masked_mae(prediction, target, null_val=null_val)

    if mode == 'finetune' or repr1 is None or repr2 is None or model is None:
        # Only prediction loss in finetune mode
        return pred_loss

    # Compute SSL losses
    temporal_loss, spatial_loss = model.compute_ssl_losses(repr1, repr2)

    if mode == 'pretrain':
        # Only SSL losses in pretrain mode
        total_loss = loss_weights[1] * temporal_loss + loss_weights[2] * spatial_loss
    else:
        # End-to-end training with all losses
        total_loss = (
                loss_weights[0] * pred_loss +
                loss_weights[1] * temporal_loss +
                loss_weights[2] * spatial_loss
        )

    return total_loss


class STSSLLoss(nn.Module):
    """
    ST-SSL Loss Module.

    Combines prediction loss with temporal and spatial contrastive losses.
    """

    def __init__(
            self,
            null_val: float = 0.0,
            loss_weights: list = None,
            mode: str = 'end2end'
    ):
        """
        Args:
            null_val: null value for masking in MAE computation
            loss_weights: weights for [pred_loss, temporal_loss, spatial_loss]
            mode: training mode ('pretrain', 'finetune', 'end2end')
        """
        super(STSSLLoss, self).__init__()
        self.null_val = null_val
        self.loss_weights = loss_weights if loss_weights is not None else [1.0, 1.0, 1.0]
        self.mode = mode

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
            repr1: torch.Tensor = None,
            repr2: torch.Tensor = None,
            model: nn.Module = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Compute combined ST-SSL loss.

        Args:
            prediction: predicted values, [B, L, N, C]
            target: ground truth values, [B, L, N, C]
            repr1: representation from view 1
            repr2: representation from view 2
            model: STSSL model for computing SSL losses

        Returns:
            total_loss: combined loss
        """
        return stssl_loss(
            prediction=prediction,
            target=target,
            repr1=repr1,
            repr2=repr2,
            model=model,
            null_val=self.null_val,
            loss_weights=self.loss_weights,
            mode=self.mode,
            **kwargs
        )
