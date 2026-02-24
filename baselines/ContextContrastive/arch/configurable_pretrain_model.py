"""
Configurable Contrastive Pre-training Model

Allows specifying augmentation strategy for ablation studies.
"""
import torch
import torch.nn as nn

from .context_aware_encoder import ContextAwareEncoder
from .augmentations import ConfigurableAugmentation


class ConfigurablePretrainModel(nn.Module):
    """
    Contrastive Pre-training Model with configurable augmentation.

    For ablation studies comparing different augmentation strategies:
    - Temporal masking only
    - Feature masking only
    - Combined augmentations
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
        # Augmentation config
        augmentation_config: dict = None,
        # Multi-resolution contrast (time downsampling factor, e.g., 3 for 15min)
        multi_resolution_factor: int = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.d_model = d_model
        self.multi_resolution_factor = max(1, int(multi_resolution_factor))

        # Context-aware encoder (temporal only for stability)
        self.encoder = ContextAwareEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            use_spatial=False,  # Temporal only for now
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Configurable augmentation
        if augmentation_config is None:
            augmentation_config = {
                'temporal_masking': {'mask_ratio': 0.15},
                'gaussian_noise': {'noise_std': 0.05},
            }
        self.augmentation = ConfigurableAugmentation(augmentation_config)

        # Dummy predictor for metric compatibility
        self.dummy_predictor = nn.Linear(d_model, output_len * output_dim)

    def _downsample_time(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample time dimension by non-overlapping mean pooling.

        Args:
            x: [B, T, N, C]
            factor: aggregation factor (e.g., 3 for 15min)
        Returns:
            x_coarse: [B, T//factor, N, C]
        """
        if factor <= 1:
            return x
        B, T, N, C = x.shape
        T_new = T // factor
        if T_new <= 0:
            return x
        x = x[:, :T_new * factor].reshape(B, T_new, factor, N, C)
        return x.mean(dim=2)

    def _pool_time(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pool over time dimension.

        Args:
            z: [B, T, N, D]
        Returns:
            z_pooled: [B, N, D]
        """
        return z.mean(dim=1)

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
        Forward pass for contrastive pre-training.

        Multi-resolution mode (factor > 1):
            - view1: fine resolution (original)
            - view2: coarse resolution (downsampled)
            - Both are time-pooled before contrast

        Same-resolution mode (factor = 1):
            - view1, view2: same resolution, different augmentations
            - Both are time-pooled before contrast

        Args:
            history_data: [B, T, N, C] - input time series

        Returns:
            dict with 'prediction', 'z1', 'z2'
        """
        B, T, N, C = history_data.shape

        if train:
            # Create two augmented views
            x_view1 = self.augmentation(history_data)
            x_view2 = self.augmentation(history_data)

            if self.multi_resolution_factor > 1:
                # Multi-resolution contrast: fine vs. coarse (downsampled)
                x_view2 = self._downsample_time(x_view2, self.multi_resolution_factor)

            # Encode both views
            z1_full = self.encoder(x_view1, edge_index)  # [B, T, N, D]
            z2_full = self.encoder(x_view2, edge_index)  # [B, T' or T, N, D]

            # Time pooling (consistent for both modes)
            z1 = self._pool_time(z1_full)  # [B, N, D]
            z2 = self._pool_time(z2_full)  # [B, N, D]

            # Project
            z1_proj = self.projector(z1)
            z2_proj = self.projector(z2)
        else:
            # No augmentation during eval
            z_full = self.encoder(history_data, edge_index)
            z_pool = self._pool_time(z_full)
            z1_proj = z2_proj = self.projector(z_pool)

        # Dummy prediction (for metric compatibility)
        pred = self.dummy_predictor(z1_proj)
        pred = pred.reshape(B, N, self.output_len, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)

        return {
            'prediction': pred,
            'z1': z1_proj,
            'z2': z2_proj,
        }

    def get_encoder(self):
        """Return the pre-trained encoder."""
        return self.encoder
