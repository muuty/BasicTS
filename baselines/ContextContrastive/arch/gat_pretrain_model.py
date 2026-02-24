"""
GAT-based Contrastive Pre-training Model

Uses SpatioTemporalEncoder with Top-K Sparse GAT for spatial encoding.
"""
import torch
import torch.nn as nn

from .spatiotemporal_encoder import SpatioTemporalEncoder
from .augmentations import ConfigurableAugmentation


class GATPretrainModel(nn.Module):
    """
    Contrastive Pre-training Model with Top-K Sparse GAT encoder.

    Architecture:
        Input → Temporal Encoder → Spatial GAT Encoder → Projection → Contrastive Loss
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int,
        output_dim: int,
        # Encoder params
        d_model: int = 64,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        spatial_layers: int = 1,
        spatial_heads: int = 4,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        # Adjacency matrix
        adj_matrix: torch.Tensor = None,
        # Augmentation config
        augmentation_config: dict = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.d_model = d_model

        # Register adjacency matrix as buffer (moves with model to GPU)
        if adj_matrix is not None:
            self.register_buffer('adj_matrix', adj_matrix)
        else:
            self.adj_matrix = None

        # Spatio-Temporal encoder with GAT
        self.encoder = SpatioTemporalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            k_neighbors=k_neighbors,
            dropout=dropout,
        )

        # Projection head for contrastive learning
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
            dict with 'prediction', 'z1', 'z2'
        """
        B, T, N, C = history_data.shape

        # Use registered adjacency matrix
        adj = self.adj_matrix

        if train:
            # Create two augmented views
            x_aug1 = self.augmentation(history_data)
            x_aug2 = self.augmentation(history_data)

            # Encode both views with GAT spatial encoder
            z1_full = self.encoder(x_aug1, adj)  # [B, T, N, D]
            z2_full = self.encoder(x_aug2, adj)  # [B, T, N, D]

            # Use last timestep for contrastive
            z1 = z1_full[:, -1, :, :]  # [B, N, D]
            z2 = z2_full[:, -1, :, :]  # [B, N, D]

            # Project
            z1_proj = self.projector(z1)
            z2_proj = self.projector(z2)
        else:
            # No augmentation during eval
            z_full = self.encoder(history_data, adj)
            z1_proj = z2_proj = self.projector(z_full[:, -1, :, :])

        # Dummy prediction for metric compatibility
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
