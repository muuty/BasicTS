"""
Linear Attention Denoising Encoder for Noise-Robust Traffic Prediction.

Architecture:
  Stage 1: Temporal Linear Attention (across timesteps per node)
  Stage 2: Spatial Linear Attention (across nodes per timestep)

Linear attention (ELU+1 kernel) avoids softmax's exponential amplification
of outlier values, making it inherently more robust to noisy inputs.

No adjacency matrix required — fully model-agnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder


class LinearAttention(nn.Module):
    """Linear attention with ELU+1 feature map.

    Computes: Attn(Q, K, V) = φ(Q) (φ(K)^T V) / (φ(Q) φ(K)^T 1)
    where φ(x) = ELU(x) + 1

    Complexity: O(N * d^2) instead of O(N^2 * d) for softmax attention.
    No exponential amplification of outlier keys.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """φ(x) = ELU(x) + 1, ensures non-negative."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] where L is sequence length (T for temporal, N for spatial)
        Returns:
            [B, L, D]
        """
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d]
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        Q = self._feature_map(Q)  # [B, H, L, d]
        K = self._feature_map(K)  # [B, H, L, d]

        # Linear attention: φ(Q) @ (φ(K)^T @ V) — O(L*d^2)
        KV = torch.einsum('bhld,bhlv->bhdv', K, V)  # [B, H, d, d]
        num = torch.einsum('bhld,bhdv->bhlv', Q, KV)  # [B, H, L, d]

        # Normalization: φ(Q) @ φ(K)^T @ 1
        K_sum = K.sum(dim=2)  # [B, H, d]
        denom = torch.einsum('bhld,bhd->bhl', Q, K_sum).unsqueeze(-1).clamp(min=1e-6)  # [B, H, L, 1]

        out = num / denom  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class AttentionBlock(nn.Module):
    """Linear attention + FFN with pre-norm residual."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@register_encoder('LinearAttentionDenoisingEncoder')
class LinearAttentionDenoisingEncoder(BaseRepresentationEncoder):
    """
    Denoising encoder using linear attention for both temporal and spatial dims.

    Stage 1: Temporal linear attention (per node, across timesteps)
    Stage 2: Spatial linear attention (per timestep, across nodes)

    No adjacency matrix needed. Linear attention avoids outlier amplification.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 64,
        n_heads: int = 4,
        temporal_layers: int = 2,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        physical_channels: List[int] = None,
        residual_connection: bool = True,
        **kwargs,
    ):
        # adj_path not needed, pop it to avoid errors
        kwargs.pop('adj_path', None)
        super().__init__(input_dim, d_model, adj_path=None, **kwargs)

        self.hidden_dim = hidden_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        self.residual_connection = residual_connection
        n_physical = len(self.physical_channels)

        # Input projection
        self.input_proj = nn.Linear(n_physical, hidden_dim)

        # Stage 1: Temporal attention blocks
        self.temporal_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(temporal_layers)
        ])

        # Stage 2: Spatial attention blocks
        self.spatial_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(spatial_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_physical)

    def encode(self, x: torch.Tensor, **kwargs):
        """
        Args:
            x: [B, T, N, D_in]
        Returns:
            [B, T, N, D_in] with denoised physical channels
        """
        B, T, N, D = x.shape

        # Extract physical channels
        x_physical = x[..., self.physical_channels]  # [B, T, N, n_phys]

        # Project to hidden dim
        h = self.input_proj(x_physical)  # [B, T, N, hidden]

        # Stage 1: Temporal attention (per node)
        # Reshape: [B*N, T, hidden]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        for block in self.temporal_blocks:
            h = block(h)
        h = h.reshape(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)  # [B, T, N, hidden]

        # Stage 2: Spatial attention (per timestep)
        # Reshape: [B*T, N, hidden]
        h = h.reshape(B * T, N, self.hidden_dim)
        for block in self.spatial_blocks:
            h = block(h)
        h = h.reshape(B, T, N, self.hidden_dim)  # [B, T, N, hidden]

        # Output projection
        correction = self.output_proj(h)  # [B, T, N, n_phys]

        if self.residual_connection:
            denoised = x_physical + correction
        else:
            denoised = correction

        # Reassemble: replace physical channels, keep tod/dow
        out = x.clone()
        for i, ch in enumerate(self.physical_channels):
            out[..., ch] = denoised[..., i]

        return out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'LinearAttentionDenoisingEncoder',
            'hidden_dim': self.hidden_dim,
            'temporal_layers': len(self.temporal_blocks),
            'spatial_layers': len(self.spatial_blocks),
            'physical_channels': self.physical_channels,
            'residual_connection': self.residual_connection,
        })
        return config
