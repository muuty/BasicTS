"""
T-Rep Encoder for Spatial-Temporal Data

T-Rep extends TS2Vec with:
1. Learnable time embeddings
2. Additional pretext tasks (temporal relation prediction)

Adapted for traffic data with shape [B, T, N, C].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from baselines.TS2Vec.arch.dilated_conv import DilatedConvEncoder


class TimeEmbedding(nn.Module):
    """
    Learnable time embedding module.

    Converts time indices to continuous embeddings that capture
    temporal relationships.
    """

    def __init__(self, embed_dim: int = 64, learnable: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.learnable = learnable

        if learnable:
            # Learnable Fourier features
            self.freq = nn.Parameter(torch.randn(embed_dim // 2) * 0.1)
            self.phase = nn.Parameter(torch.zeros(embed_dim // 2))

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_indices: [B, T] or [B, T, 1] - normalized time indices [0, 1]

        Returns:
            Time embeddings [B, T, embed_dim]
        """
        if time_indices.dim() == 3:
            time_indices = time_indices.squeeze(-1)

        # Scale to [0, 2*pi]
        t = time_indices.unsqueeze(-1) * 2 * math.pi  # [B, T, 1]

        if self.learnable:
            # Learnable Fourier features
            freq = self.freq.unsqueeze(0).unsqueeze(0)  # [1, 1, D/2]
            phase = self.phase.unsqueeze(0).unsqueeze(0)  # [1, 1, D/2]

            sin_embed = torch.sin(t * freq + phase)
            cos_embed = torch.cos(t * freq + phase)
            embed = torch.cat([sin_embed, cos_embed], dim=-1)
        else:
            # Fixed positional encoding
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2, device=time_indices.device).float()
                * (-math.log(10000.0) / self.embed_dim)
            )
            sin_embed = torch.sin(t * div_term)
            cos_embed = torch.cos(t * div_term)
            embed = torch.cat([sin_embed, cos_embed], dim=-1)

        return embed


class TRepEncoder(nn.Module):
    """
    T-Rep Encoder for spatial-temporal time series.

    Extends TS2Vec encoder with learnable time embeddings.

    Input: [B, T, N, C]
    Output: [B, T, N, D], [B, T, N, time_embed_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 10,
        time_embed_dim: int = 64,
        mask_mode: str = 'binomial',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.mask_mode = mask_mode

        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Dilated conv encoder
        self.feature_extractor = DilatedConvEncoder(
            hidden_dim,
            [hidden_dim] * depth + [output_dim],
            kernel_size=3
        )

        # Time embedding
        self.time_embedding = TimeEmbedding(embed_dim=time_embed_dim, learnable=True)

        self.repr_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_indices: torch.Tensor = None,
        mask: str = None
    ) -> tuple:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, N, C]
            time_indices: Optional time indices [B, T] (normalized to [0, 1])
            mask: Masking mode

        Returns:
            Tuple of (representations [B, T, N, D], time_embeddings [B, T, N, time_embed_dim])
        """
        B, T, N, C = x.shape

        # Generate time indices if not provided
        if time_indices is None:
            time_indices = torch.linspace(0, 1, T, device=x.device)
            time_indices = time_indices.unsqueeze(0).expand(B, -1)

        # Compute time embeddings: [B, T, time_embed_dim]
        time_embed = self.time_embedding(time_indices)

        # Handle NaN values
        nan_mask = ~x.isnan().any(dim=-1)
        x = torch.nan_to_num(x, nan=0.0)

        # Reshape to process all nodes: [B*N, T, C]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        nan_mask_flat = nan_mask.permute(0, 2, 1).reshape(B * N, T)

        # Input projection
        x = self.input_fc(x)

        # Generate and apply mask
        if mask is None:
            mask = self.mask_mode if self.training else 'all_true'

        if mask == 'binomial':
            mask_tensor = torch.from_numpy(
                np.random.binomial(1, 0.5, size=(B * N, T))
            ).to(torch.bool).to(x.device)
        elif mask == 'all_true':
            mask_tensor = x.new_full((B * N, T), True, dtype=torch.bool)
        else:
            mask_tensor = x.new_full((B * N, T), True, dtype=torch.bool)

        mask_tensor = mask_tensor & nan_mask_flat
        x[~mask_tensor] = 0

        # Conv encoder
        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        x = x.transpose(1, 2)  # [B*N, T, D]

        # Reshape back: [B, N, T, D] -> [B, T, N, D]
        x = x.reshape(B, N, T, self.output_dim).permute(0, 2, 1, 3)

        # Expand time embeddings for each node: [B, T, N, time_embed_dim]
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, N, -1)

        return x, time_embed


class TemporalRelationHead(nn.Module):
    """
    Prediction head for temporal relation tasks.

    Predicts the temporal distance/relationship between two timesteps.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: Representations [B, D]

        Returns:
            Temporal relation prediction [B, output_dim]
        """
        z = torch.cat([z1, z2], dim=-1)
        return self.net(z)


class TRepPretrainModel(nn.Module):
    """
    T-Rep Pre-training Model for spatial-temporal forecasting.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int,
        output_dim: int = 1,
        d_model: int = 64,
        hidden_dim: int = 64,
        depth: int = 10,
        time_embed_dim: int = 64,
        mask_mode: str = 'binomial',
        dropout: float = 0.1,
        temporal_unit: int = 0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim
        self.temporal_unit = temporal_unit

        # T-Rep Encoder
        self.encoder = TRepEncoder(
            input_dim=input_dim,
            output_dim=d_model,
            hidden_dim=hidden_dim,
            depth=depth,
            time_embed_dim=time_embed_dim,
            mask_mode=mask_mode,
            dropout=dropout,
        )

        # Temporal relation prediction head
        self.temporal_relation_head = TemporalRelationHead(
            input_dim=d_model,
            hidden_dim=128,
            output_dim=1
        )

        # Conditional prediction head (predict representation given time embedding)
        self.cond_pred_head = nn.Sequential(
            nn.Linear(d_model + time_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        # Dummy predictor for metric compatibility
        self.dummy_predictor = nn.Linear(d_model, output_len * output_dim)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> dict:
        """
        Forward pass for pre-training.
        """
        B, T, N, C = history_data.shape

        # Generate time indices (normalized)
        time_indices = torch.linspace(0, 1, T, device=history_data.device)
        time_indices = time_indices.unsqueeze(0).expand(B, -1)

        if train:
            # Random cropping for two views
            crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=T + 1)
            crop_left = np.random.randint(T - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=T + 1)

            # View 1
            x1 = history_data[:, crop_eleft:crop_right, :, :]
            t1 = time_indices[:, crop_eleft:crop_right]
            z1_full, tau1_full = self.encoder(x1, t1)
            z1 = z1_full[:, -(crop_right - crop_left):, :, :]
            tau1 = tau1_full[:, -(crop_right - crop_left):, :, :]

            # View 2
            x2 = history_data[:, crop_left:crop_eright, :, :]
            t2 = time_indices[:, crop_left:crop_eright]
            z2_full, tau2_full = self.encoder(x2, t2)
            z2 = z2_full[:, :crop_l, :, :]
            tau2 = tau2_full[:, :crop_l, :, :]
        else:
            z1, tau1 = self.encoder(history_data, time_indices)
            z2, tau2 = z1, tau1
            crop_l = T

        # Dummy prediction
        z_last = z1[:, -1, :, :]
        pred = self.dummy_predictor(z_last)
        pred = pred.reshape(B, N, self.output_len, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)

        return {
            'prediction': pred,
            'z1': z1,
            'z2': z2,
            'tau1': tau1,
            'tau2': tau2,
            'crop_l': crop_l,
        }

    def get_encoder(self):
        """Return the pre-trained encoder."""
        return self.encoder
