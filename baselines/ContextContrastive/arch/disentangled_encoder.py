"""
Disentangled Encoder: Separates context-aware and self-intrinsic representations.

z_context: Captures spatial/temporal correlations with neighbors (Transformer)
z_self: Captures node-specific intrinsic properties (MLP)

Usage:
    encoder = DisentangledEncoder(
        input_dim=3,
        d_model=64,
        num_layers=2,
        nhead=4,
        use_orthogonality=True,  # v2
    )
    z_context, z_self = encoder(x)  # x: [B, T, N, C]
"""
import torch
import torch.nn as nn
import math

from .base_encoder import register_encoder, BaseRepresentationEncoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, ...] or [B*N, T, D]
        return x + self.pe[:, :x.size(1)]


class ContextEncoder(nn.Module):
    """Context-aware encoder using Transformer.

    Captures spatial and temporal correlations.
    """

    def __init__(self, input_dim: int, d_model: int, num_layers: int = 2,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] input tensor
        Returns:
            z_context: [B, T, N, d_model] context-aware representation
        """
        B, T, N, C = x.shape

        # Project input
        x = self.input_proj(x)  # [B, T, N, d_model]

        # Reshape for transformer: treat each node's time series independently
        # [B, T, N, D] -> [B*N, T, D]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)  # [B*N, T, D]

        # Reshape back: [B*N, T, D] -> [B, T, N, D]
        x = x.reshape(B, N, T, -1).permute(0, 2, 1, 3)

        return x


class SelfEncoder(nn.Module):
    """Self-intrinsic encoder using MLP.

    Captures node-specific properties independent of neighbors.
    """

    def __init__(self, input_dim: int, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] input tensor
        Returns:
            z_self: [B, T, N, d_model] self-intrinsic representation
        """
        return self.mlp(x)


@register_encoder('DisentangledEncoder')
class DisentangledEncoder(BaseRepresentationEncoder):
    """Disentangled encoder producing context-aware and self-intrinsic representations.

    Args:
        input_dim: Input feature dimension
        d_model: Hidden dimension for both encoders
        num_layers: Number of transformer layers for context encoder
        nhead: Number of attention heads
        dropout: Dropout rate
        fusion: How to combine representations ('concat', 'sum', 'gate')
        use_orthogonality: Whether to compute orthogonality loss
        ortho_weight: Weight for orthogonality loss
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        fusion: str = 'concat',
        use_orthogonality: bool = False,
        ortho_weight: float = 0.1,
        **kwargs  # Accept extra args from base class
    ):
        super().__init__(input_dim=input_dim, d_model=d_model, **kwargs)
        self.d_model = d_model
        self.fusion = fusion
        self.use_orthogonality = use_orthogonality
        self.ortho_weight = ortho_weight

        # Context encoder (Transformer-based)
        self.context_encoder = ContextEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        )

        # Self encoder (MLP-based)
        self.self_encoder = SelfEncoder(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout
        )

        # Fusion layer if needed
        if fusion == 'concat':
            self._output_dim = d_model * 2
        elif fusion == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self._output_dim = d_model
        elif fusion in ('context_only', 'self_only'):
            self._output_dim = d_model
        else:  # sum
            self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder (overrides base class)."""
        return self._output_dim

    def forward(self, x: torch.Tensor, return_disentangled: bool = False):
        """
        Args:
            x: [B, T, N, C] input tensor
            return_disentangled: If True, return (z_context, z_self) separately
        Returns:
            If return_disentangled:
                z_context: [B, T, N, d_model]
                z_self: [B, T, N, d_model]
            Else:
                z_fused: [B, T, N, output_dim]
        """
        z_context = self.context_encoder(x)  # [B, T, N, d_model]
        z_self = self.self_encoder(x)        # [B, T, N, d_model]

        if return_disentangled:
            return z_context, z_self

        # Fuse representations
        if self.fusion == 'concat':
            z_fused = torch.cat([z_context, z_self], dim=-1)
        elif self.fusion == 'gate':
            gate = self.gate(torch.cat([z_context, z_self], dim=-1))
            z_fused = gate * z_context + (1 - gate) * z_self
        elif self.fusion == 'context_only':
            z_fused = z_context
        elif self.fusion == 'self_only':
            z_fused = z_self
        else:  # sum
            z_fused = z_context + z_self

        return z_fused

    def compute_orthogonality_loss(self, z_context: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality loss to encourage disentanglement.

        Loss = ||z_context^T @ z_self||^2 (should be 0 if orthogonal)
        """
        # Normalize
        z_context_norm = z_context / (z_context.norm(dim=-1, keepdim=True) + 1e-8)
        z_self_norm = z_self / (z_self.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute cosine similarity and penalize non-zero values
        cos_sim = (z_context_norm * z_self_norm).sum(dim=-1)  # [B, T, N]
        ortho_loss = cos_sim.pow(2).mean()

        return ortho_loss * self.ortho_weight

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode for downstream tasks (returns fused representation)."""
        return self.forward(x, return_disentangled=False)

    def encode_disentangled(self, x: torch.Tensor) -> tuple:
        """Encode and return both representations separately."""
        return self.forward(x, return_disentangled=True)
