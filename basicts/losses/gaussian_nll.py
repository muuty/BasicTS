import torch
import numpy as np


def hybrid_nll_loss(prediction, target, sigma, epoch,
                    null_val=np.nan, alpha=0.5,
                    warmup_epochs=0, decay_epochs=1, alpha_min=0.5):
    """MAE + Gaussian NLL hybrid loss with optional alpha curriculum.

    L = alpha * MAE(mu, y) + (1 - alpha) * NLL(mu, sigma, y)
    NLL = 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]

    Alpha curriculum: stays at 1.0 during warmup, then linearly decays to alpha_min.
    For fixed alpha, set warmup_epochs=0 and alpha_min=desired_alpha.

    Args:
        prediction: mu, (B, T, N, 1)
        target: ground truth, (B, T, N, 1)
        sigma: predicted std, (B, T, N, 1), positive
        epoch: current epoch (tensor)
        null_val: value to mask
        alpha: initial alpha (only used if warmup > 0)
        warmup_epochs: epochs of pure MAE (alpha=1.0)
        decay_epochs: epochs over which alpha decays from 1.0 to alpha_min
        alpha_min: minimum alpha value
    """
    # Mask
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val, device=target.device).expand_as(target), atol=eps, rtol=0.)
    mask = mask.float()
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    # Alpha schedule
    ep = epoch.item() if torch.is_tensor(epoch) else epoch
    if ep < warmup_epochs:
        a = 1.0
    elif decay_epochs > 0:
        progress = min((ep - warmup_epochs) / decay_epochs, 1.0)
        a = 1.0 - progress * (1.0 - alpha_min)
    else:
        a = alpha_min

    # MAE component
    ae = torch.abs(prediction - target) * mask
    mae = ae.sum() / mask_sum

    # NLL component: 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]
    var = sigma ** 2
    nll = 0.5 * (torch.log(var) + (prediction - target) ** 2 / var) * mask
    nll = nll.sum() / mask_sum

    loss = a * mae + (1.0 - a) * nll
    return loss
