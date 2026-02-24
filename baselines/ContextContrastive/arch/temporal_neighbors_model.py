"""
Temporal Neighbors Contrastive Pre-training Model

Unlike the original ContrastivePretrainModel that only uses the last timestep,
this model outputs all timesteps for hierarchical temporal contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_aware_encoder import ContextAwareEncoder


class TemporalAugmentation(nn.Module):
    """
    Enhanced augmentation module for temporal contrastive learning.

    Includes standard augmentations plus temporal-aware augmentations.
    """

    def __init__(
        self,
        temporal_mask_ratio: float = 0.15,
        feature_dropout_ratio: float = 0.1,
        noise_std: float = 0.05,
        time_shift_max: int = 2,
    ):
        super().__init__()
        self.temporal_mask_ratio = temporal_mask_ratio
        self.feature_dropout_ratio = feature_dropout_ratio
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max

    def temporal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask some timesteps."""
        B, T, N, C = x.shape
        mask = torch.rand(B, T, 1, 1, device=x.device) > self.temporal_mask_ratio
        return x * mask.float()

    def feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop some features."""
        B, T, N, C = x.shape
        mask = torch.rand(B, 1, 1, C, device=x.device) > self.feature_dropout_ratio
        return x * mask.float()

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly shift time series (circular)."""
        if self.time_shift_max == 0:
            return x
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        x = self.temporal_mask(x)
        x = self.feature_dropout(x)
        x = self.gaussian_noise(x)
        # Note: time_shift disabled by default to preserve temporal alignment
        # x = self.time_shift(x)
        return x


class TemporalNeighborsModel(nn.Module):
    """
    Temporal Neighbors Contrastive Pre-training Model

    Key differences from ContrastivePretrainModel:
    1. Outputs all timesteps (not just last) for hierarchical temporal learning
    2. Uses enhanced temporal augmentations
    3. Designed for temporal_neighbors_loss

    Architecture:
    - ContextAwareEncoder: temporal encoding
    - Projector: projection head for contrastive learning
    - TemporalAugmentation: data augmentation
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        # Augmentation params
        temporal_mask_ratio: float = 0.15,
        feature_dropout_ratio: float = 0.1,
        noise_std: float = 0.05,
        **kwargs
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model

        # Context-aware encoder (temporal only)
        self.encoder = ContextAwareEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Augmentation
        self.augmentation = TemporalAugmentation(
            temporal_mask_ratio=temporal_mask_ratio,
            feature_dropout_ratio=feature_dropout_ratio,
            noise_std=noise_std,
        )

        # Dummy predictor for metric compatibility
        self.dummy_predictor = nn.Linear(d_model, output_len * output_dim)
        self.output_dim = output_dim

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> dict:
        """
        Args:
            history_data: [B, T, N, C] - input time series

        Returns:
            dict with:
                - 'prediction': dummy prediction for metrics
                - 'z1': [B, T, N, D] - first view (all timesteps)
                - 'z2': [B, T, N, D] - second view (all timesteps)
        """
        B, T, N, C = history_data.shape

        if train:
            # Create two augmented views
            x_aug1 = self.augmentation(history_data)
            x_aug2 = self.augmentation(history_data)

            # Encode both views - get ALL timesteps
            z1_full = self.encoder(x_aug1, edge_index)  # [B, T, N, D]
            z2_full = self.encoder(x_aug2, edge_index)  # [B, T, N, D]

            # Project ALL timesteps
            z1_proj = self.projector(z1_full)  # [B, T, N, D]
            z2_proj = self.projector(z2_full)  # [B, T, N, D]
        else:
            # No augmentation during eval
            z_full = self.encoder(history_data, edge_index)
            z1_proj = z2_proj = self.projector(z_full)

        # Dummy prediction using last timestep
        z_last = z1_proj[:, -1, :, :]  # [B, N, D]
        pred = self.dummy_predictor(z_last)  # [B, N, output_len * output_dim]
        pred = pred.reshape(B, N, self.output_len, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)  # [B, T', N, C]

        return {
            'prediction': pred,
            'z1': z1_proj,  # [B, T, N, D] - all timesteps
            'z2': z2_proj,  # [B, T, N, D] - all timesteps
        }
