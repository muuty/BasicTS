import torch
import numpy as np

from basicts.metrics import masked_mae


def spillover_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_noisy: torch.Tensor = None,
    corrupt_mask: torch.Tensor = None,
    null_val: float = 0.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """Loss for noise-augmented spillover correction training.

    L = L_clean + β · L_noisy

    - L_clean: MAE on all nodes (clean forward prediction)
    - L_noisy: MAE on non-corrupt nodes only (noisy forward prediction)

    During validation/test, prediction_noisy and corrupt_mask are None,
    so only L_clean is computed.

    Args:
        prediction: (B, T, N, 1) clean prediction (μ_c + δ_c)
        target: (B, T, N, 1) ground truth
        prediction_noisy: (B, T, N, 1) noisy prediction (μ_n + δ_n), None at val/test
        corrupt_mask: (B, N) boolean mask, True for corrupted nodes, None at val/test
        null_val: value to mask in target (default 0.0)
        beta: weight for L_noisy (default 0.5)

    Returns:
        Scalar loss tensor.
    """
    L_clean = masked_mae(prediction, target, null_val)

    if prediction_noisy is None or corrupt_mask is None:
        return L_clean

    # Expand corrupt_mask: (B, N) → (B, 1, N, 1) → broadcast to (B, T, N, 1)
    node_mask = ~corrupt_mask  # (B, N) — True for clean nodes
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
    node_mask = node_mask.expand_as(target).float()   # (B, T, N, 1)

    # Also apply null_val masking
    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(
            target,
            torch.tensor(null_val).expand_as(target).to(target.device),
            atol=eps, rtol=0.0,
        )
    null_mask = null_mask.float()

    # Combined mask: non-corrupt AND non-null
    combined_mask = node_mask * null_mask
    combined_mask /= torch.mean(combined_mask).clamp(min=1e-8)
    combined_mask = torch.nan_to_num(combined_mask)

    loss_noisy = torch.abs(prediction_noisy - target) * combined_mask
    loss_noisy = torch.nan_to_num(loss_noisy)
    L_noisy = torch.mean(loss_noisy)

    return L_clean + beta * L_noisy
