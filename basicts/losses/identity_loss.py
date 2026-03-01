"""Identity-regularized loss for noise-resilient prediction.

L = L_pred + passthrough_loss

- L_pred: standard masked MAE on predictions
- passthrough_loss: pre-weighted ||encoder(clean) - clean|| on physical channels
  Forces encoder to be identity on clean inputs (CycleGAN identity loss principle).
  Weight is applied in the runner, not here.
"""

import torch
from basicts.metrics import masked_mae


def identity_regularized_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    passthrough_loss: torch.Tensor = None,
    null_val: float = 0.0,
) -> torch.Tensor:
    L_pred = masked_mae(prediction, target, null_val)
    if passthrough_loss is None:
        return L_pred
    return L_pred + passthrough_loss
