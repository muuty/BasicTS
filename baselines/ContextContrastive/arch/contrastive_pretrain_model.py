"""
Contrastive Pre-training Model (Stage 1)

Uses augmentation-based contrastive learning to pre-train the ContextAwareEncoder.
SimCLR-style: same sample with different augmentations = positive pair.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_aware_encoder import ContextAwareEncoder


class Augmentation(nn.Module):
    """Augmentation module for contrastive learning."""

    def __init__(
        self,
        temporal_mask_ratio: float = 0.15,
        feature_dropout_ratio: float = 0.1,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.temporal_mask_ratio = temporal_mask_ratio
        self.feature_dropout_ratio = feature_dropout_ratio
        self.noise_std = noise_std

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        x = self.temporal_mask(x)
        x = self.feature_dropout(x)
        x = self.gaussian_noise(x)
        return x


class ContrastivePretrainModel(nn.Module):
    """
    Stage 1: Contrastive Pre-training Model

    Architecture:
    - ContextAwareEncoder: temporal + spatial encoding
    - Projector: projection head for contrastive learning
    - Augmentation: data augmentation for creating views

    Returns z1, z2 (two augmented views) for contrastive loss.
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
        use_spatial: bool = True,
        # Augmentation params
        temporal_mask_ratio: float = 0.15,
        feature_dropout_ratio: float = 0.1,
        noise_std: float = 0.05,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model

        # Context-aware encoder
        self.encoder = ContextAwareEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            use_spatial=use_spatial,
        )

        # Projection head (for contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Augmentation
        self.augmentation = Augmentation(
            temporal_mask_ratio=temporal_mask_ratio,
            feature_dropout_ratio=feature_dropout_ratio,
            noise_std=noise_std,
        )

        # Dummy predictor (for compatibility with basicts loss calculation)
        self.dummy_predictor = nn.Linear(d_model, output_len * output_dim)
        self.output_len = output_len
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
            future_data: [B, T', N, C] - target (not used in contrastive)
            edge_index: [2, E] - edge indices for spatial graph
            batch_seen, epoch, train: for compatibility

        Returns:
            dict with 'prediction', 'z1', 'z2'
        """
        B, T, N, C = history_data.shape

        if train:
            # Create two augmented views
            x_aug1 = self.augmentation(history_data)
            x_aug2 = self.augmentation(history_data)

            # Encode both views
            z1_full = self.encoder(x_aug1, edge_index)  # [B, T, N, D]
            z2_full = self.encoder(x_aug2, edge_index)  # [B, T, N, D]

            # Use last timestep for contrastive
            z1 = z1_full[:, -1, :, :]  # [B, N, D]
            z2 = z2_full[:, -1, :, :]  # [B, N, D]

            # Project
            z1_proj = self.projector(z1)  # [B, N, D]
            z2_proj = self.projector(z2)  # [B, N, D]
        else:
            # No augmentation during eval
            z_full = self.encoder(history_data, edge_index)
            z1_proj = z2_proj = self.projector(z_full[:, -1, :, :])

        # Dummy prediction (for metric compatibility)
        pred = self.dummy_predictor(z1_proj)  # [B, N, output_len * output_dim]
        pred = pred.reshape(B, N, self.output_len, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)  # [B, T', N, C]

        return {
            'prediction': pred,
            'z1': z1_proj,
            'z2': z2_proj,
        }
