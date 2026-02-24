"""
Temporal-Aware Contrastive Pre-training Model

Extends ConfigurablePretrainModel to pass temporal context (tod, dow)
to the loss function for temporal-aware negative sampling.
"""
import torch
import torch.nn as nn

from .context_aware_encoder import ContextAwareEncoder
from .augmentations import ConfigurableAugmentation


class TemporalAwarePretrainModel(nn.Module):
    """
    Contrastive Pre-training Model with temporal context awareness.

    Key difference from ConfigurablePretrainModel:
    - Extracts tod/dow from input features
    - Passes them to forward output for temporal-aware contrastive loss
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
        # Feature indices for temporal context
        tod_feature_idx: int = 1,  # Index of time_of_day in input features
        dow_feature_idx: int = 2,  # Index of day_of_week in input features
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.d_model = d_model
        self.tod_feature_idx = tod_feature_idx
        self.dow_feature_idx = dow_feature_idx

        # Context-aware encoder (temporal only for stability)
        self.encoder = ContextAwareEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            use_spatial=False,
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
                'feature_masking': {'mask_ratio': 0.3},
            }
        self.augmentation = ConfigurableAugmentation(augmentation_config)

        # Dummy predictor for metric compatibility
        self.dummy_predictor = nn.Linear(d_model, output_len * output_dim)

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
                          C should include [flow, tod, dow, ...]

        Returns:
            dict with 'prediction', 'z1', 'z2', 'tod', 'dow'
        """
        B, T, N, C = history_data.shape

        # Extract temporal context from input (use last timestep, first node)
        # tod/dow should be same across all nodes at each timestep
        tod = None
        dow = None

        if C > self.tod_feature_idx:
            # Extract tod: [B, T, N, 1] -> [B] (last timestep, first node)
            tod_raw = history_data[:, -1, 0, self.tod_feature_idx]
            # Denormalize if needed (assuming normalized to [0, 1] range)
            # Original tod is 0-287 (5-min intervals in a day)
            tod = (tod_raw * 288).long().clamp(0, 287)

        if C > self.dow_feature_idx:
            # Extract dow: [B, T, N, 1] -> [B] (last timestep, first node)
            dow_raw = history_data[:, -1, 0, self.dow_feature_idx]
            # Denormalize if needed (assuming normalized to [0, 1] range)
            # Original dow is 0-6
            dow = (dow_raw * 7).long().clamp(0, 6)

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
            z1_proj = self.projector(z1)
            z2_proj = self.projector(z2)
        else:
            # No augmentation during eval
            z_full = self.encoder(history_data, edge_index)
            z1_proj = z2_proj = self.projector(z_full[:, -1, :, :])

        # Dummy prediction
        pred = self.dummy_predictor(z1_proj)
        pred = pred.reshape(B, N, self.output_len, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)

        return {
            'prediction': pred,
            'z1': z1_proj,
            'z2': z2_proj,
            'tod': tod,
            'dow': dow,
        }

    def get_encoder(self):
        """Return the pre-trained encoder."""
        return self.encoder
