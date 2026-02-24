"""
Context-Aware Encoder with Temporal modeling.

Architecture:
- Temporal Encoder: Self-attention over time (causal)
- Spatial context is handled by downstream backbone (STAEformer, STGCN, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Transformer-based temporal encoder with causal masking."""

    def __init__(self, c_in: int, d_model: int, num_layers: int = 2, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(c_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            z: [B, T, N, D] - encoded representations
        """
        B, T, N, C = x.shape

        # Reshape: treat each node independently
        # [B, T, N, C] -> [B*N, T, C]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # Project to d_model
        x = self.input_proj(x)  # [B*N, T, D]

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        # Encode with causal attention
        x = self.transformer(x, mask=causal_mask, is_causal=False)  # [B*N, T, D]

        # Reshape back: [B*N, T, D] -> [B, T, N, D]
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        return x


class ContextAwareEncoder(nn.Module):
    """
    Context-Aware Encoder with temporal modeling.

    Spatial context is delegated to downstream backbone models.
    This encoder focuses on learning good temporal representations
    through contrastive learning.

    Output: [B, T, N, D] encoded representations
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        **kwargs  # Accept but ignore spatial-related args for backward compatibility
    ):
        super().__init__()
        self.d_model = d_model

        # Temporal encoder only
        self.temporal_encoder = TemporalEncoder(
            c_in=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
            edge_index: ignored (for API compatibility)
        Returns:
            z: [B, T, N, D] - context-aware encoded representations
        """
        # Pure temporal encoding
        return self.temporal_encoder(x)
