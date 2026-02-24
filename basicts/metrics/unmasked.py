import torch


def unmasked_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error without any masking. All values contribute to the loss."""
    return torch.mean(torch.abs(prediction - target))


def unmasked_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error without any masking. All values contribute to the loss."""
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def unmasked_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error without any masking. All values contribute to the loss."""
    return torch.mean((prediction - target) ** 2)
