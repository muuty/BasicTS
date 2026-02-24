"""
TS2Vec Encoder for Spatial-Temporal Data

Adapted from the original TS2Vec to handle traffic data with shape [B, T, N, C].
Processes each node independently through the temporal encoder.
"""
import torch
import torch.nn as nn
import numpy as np

from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B: int, T: int, n: int = 5, l: float = 0.1) -> torch.Tensor:
    """Generate continuous masking pattern."""
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B: int, T: int, p: float = 0.5) -> torch.Tensor:
    """Generate binomial masking pattern."""
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TS2VecEncoder(nn.Module):
    """
    TS2Vec Encoder for spatial-temporal time series.

    Processes each node independently to learn temporal representations,
    then outputs timestamp-level representations for each node.

    Input: [B, T, N, C]
    Output: [B, T, N, D]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 10,
        mask_mode: str = 'binomial',
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input features per node.
            output_dim: Representation dimension.
            hidden_dim: Hidden dimension of encoder.
            depth: Number of dilated conv blocks.
            mask_mode: Masking strategy ('binomial', 'continuous', 'all_true').
            dropout: Dropout rate.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mask_mode = mask_mode

        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Dilated conv encoder
        self.feature_extractor = DilatedConvEncoder(
            hidden_dim,
            [hidden_dim] * depth + [output_dim],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: str = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, N, C]
            mask: Masking mode (optional, uses self.mask_mode if None)

        Returns:
            Representations [B, T, N, D]
        """
        B, T, N, C = x.shape

        # Handle NaN values
        nan_mask = ~x.isnan().any(dim=-1)  # [B, T, N]
        x = torch.nan_to_num(x, nan=0.0)

        # Reshape to process all nodes together: [B*N, T, C]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        nan_mask = nan_mask.permute(0, 2, 1).reshape(B * N, T)

        # Input projection: [B*N, T, hidden_dim]
        x = self.input_fc(x)

        # Generate and apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask_tensor = generate_binomial_mask(B * N, T).to(x.device)
        elif mask == 'continuous':
            mask_tensor = generate_continuous_mask(B * N, T).to(x.device)
        elif mask == 'all_true':
            mask_tensor = x.new_full((B * N, T), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask_tensor = x.new_full((B * N, T), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask_tensor = x.new_full((B * N, T), True, dtype=torch.bool)
            mask_tensor[:, -1] = False
        else:
            mask_tensor = x.new_full((B * N, T), True, dtype=torch.bool)

        mask_tensor = mask_tensor & nan_mask
        x[~mask_tensor] = 0

        # Conv encoder: [B*N, T, hidden_dim] -> [B*N, hidden_dim, T] -> [B*N, output_dim, T]
        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        x = x.transpose(1, 2)  # [B*N, T, output_dim]

        # Reshape back: [B, N, T, D] -> [B, T, N, D]
        x = x.reshape(B, N, T, self.output_dim).permute(0, 2, 1, 3)

        return x


class TS2VecPretrainModel(nn.Module):
    """
    TS2Vec Pre-training Model for spatial-temporal forecasting.

    Wraps the encoder and provides augmentation + projection head
    for contrastive pre-training.
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
        self.temporal_unit = temporal_unit

        # TS2Vec Encoder
        self.encoder = TS2VecEncoder(
            input_dim=input_dim,
            output_dim=d_model,
            hidden_dim=hidden_dim,
            depth=depth,
            mask_mode=mask_mode,
            dropout=dropout,
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

        Args:
            history_data: [B, T, N, C]

        Returns:
            dict with 'prediction', 'z1', 'z2', 'crop_l'
        """
        B, T, N, C = history_data.shape

        if train:
            # Random cropping for two views (TS2Vec augmentation)
            crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=T + 1)
            crop_left = np.random.randint(T - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=T + 1)

            # View 1: [crop_eleft, crop_right)
            x1 = history_data[:, crop_eleft:crop_right, :, :]
            z1_full = self.encoder(x1)  # [B, crop_right-crop_eleft, N, D]
            z1 = z1_full[:, -(crop_right - crop_left):, :, :]  # [B, crop_l, N, D]

            # View 2: [crop_left, crop_eright)
            x2 = history_data[:, crop_left:crop_eright, :, :]
            z2_full = self.encoder(x2)  # [B, crop_eright-crop_left, N, D]
            z2 = z2_full[:, :crop_l, :, :]  # [B, crop_l, N, D]
        else:
            z1 = z2 = self.encoder(history_data)
            crop_l = T

        # Dummy prediction for metrics
        z_last = z1[:, -1, :, :]  # [B, N, D]
        pred = self.dummy_predictor(z_last)  # [B, N, output_len * output_dim]
        pred = pred.reshape(B, N, self.output_len, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)  # [B, output_len, N, output_dim]

        return {
            'prediction': pred,
            'z1': z1,
            'z2': z2,
            'crop_l': crop_l,
        }

    def get_encoder(self):
        """Return the pre-trained encoder."""
        return self.encoder
