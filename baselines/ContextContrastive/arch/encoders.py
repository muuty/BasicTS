"""
Representation Encoder Implementations.

All encoders implement BaseRepresentationEncoder interface:
    Input:  [B, T, N, D]
    Output: [B, T, N, H]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .base_encoder import BaseRepresentationEncoder, register_encoder


# =============================================================================
# Transformer-based Encoder
# =============================================================================

@register_encoder('TransformerEncoder')
class TransformerEncoder(BaseRepresentationEncoder):
    """
    Transformer-based temporal encoder.

    Uses self-attention to capture temporal dependencies.
    Previously known as ContextAwareEncoder.
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        **kwargs
    ):
        super().__init__(input_dim, d_model, **kwargs)

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, N, D]

        Returns:
            embedding: [B, T, N, H]
        """
        B, T, N, D = x.shape

        # Reshape to process each node independently: [B*N, T, D]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Reshape back: [B, T, N, H]
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'TransformerEncoder',
            'num_layers': len(self.transformer.layers),
            'nhead': self.transformer.layers[0].self_attn.num_heads,
        })
        return config

    def _extract_encoder_weights(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract encoder weights, handling legacy ContextAwareEncoder format.

        Legacy format: encoder.temporal_encoder.xxx -> xxx
        New format: encoder.xxx -> xxx
        """
        encoder_state_dict = {}

        # Priority order: more specific prefix first
        prefix_mappings = [
            'encoder.temporal_encoder.',  # Legacy ContextAwareEncoder
            'encoder.',                    # Standard format
        ]

        for key, value in state_dict.items():
            for prefix in prefix_mappings:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    encoder_state_dict[new_key] = value
                    break

        return encoder_state_dict


# =============================================================================
# Dilated Convolution Encoder (TS2Vec style)
# =============================================================================

@register_encoder('DilatedConvEncoder')
class DilatedConvEncoder(BaseRepresentationEncoder):
    """
    Dilated convolution encoder for temporal representation learning.

    Uses exponentially dilated convolutions to capture multi-scale temporal patterns.
    Based on TS2Vec architecture.
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        hidden_dim: int = 64,
        depth: int = 10,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(input_dim, d_model, **kwargs)

        self.hidden_dim = hidden_dim
        self.depth = depth

        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Dilated conv layers
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.conv_layers.append(
                DilatedConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_fc = nn.Linear(hidden_dim, d_model)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, N, D]

        Returns:
            embedding: [B, T, N, H]
        """
        B, T, N, D = x.shape

        # Reshape: [B*N, T, D]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Input projection
        x = self.input_fc(x)  # [B*N, T, hidden]

        # Transpose for conv: [B*N, hidden, T]
        x = x.transpose(1, 2)

        # Dilated convolutions with residual connections
        for conv in self.conv_layers:
            x = conv(x)

        # Transpose back: [B*N, T, hidden]
        x = x.transpose(1, 2)

        # Output projection
        x = self.output_fc(x)  # [B*N, T, d_model]

        # Reshape: [B, T, N, H]
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'DilatedConvEncoder',
            'hidden_dim': self.hidden_dim,
            'depth': self.depth,
        })
        return config


class DilatedConvBlock(nn.Module):
    """Single dilated convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            output: [B, C, T]
        """
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


# =============================================================================
# Spatio-Temporal Encoder (Transformer + GAT)
# =============================================================================

@register_encoder('SpatioTemporalEncoder')
class SpatioTemporalEncoder(BaseRepresentationEncoder):
    """
    Spatio-Temporal encoder combining temporal and spatial processing.

    Uses Transformer for temporal encoding and Graph Attention for spatial encoding.
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        spatial_layers: int = 1,
        spatial_heads: int = 4,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(input_dim, d_model, **kwargs)

        self.k_neighbors = k_neighbors

        # Temporal encoder
        self.temporal_encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=temporal_layers,
            nhead=temporal_heads,
            dropout=dropout,
        )

        # Spatial GAT layers
        self.spatial_layers = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        for _ in range(spatial_layers):
            self.spatial_layers.append(
                GraphAttentionLayer(d_model, d_model, spatial_heads, dropout)
            )
            self.spatial_norms.append(nn.LayerNorm(d_model))

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, N, D]

        Returns:
            embedding: [B, T, N, H]
        """
        B, T, N, D = x.shape

        # Temporal encoding
        x = self.temporal_encoder.encode(x)  # [B, T, N, H]

        # Spatial encoding (if adjacency available)
        if self.adj is not None:
            # Sparsify adjacency to top-k neighbors
            adj_sparse = self._sparsify_adj(self.adj)

            # Apply GAT layers per timestep
            outputs = []
            for t in range(T):
                h = x[:, t, :, :]  # [B, N, H]
                for gat, norm in zip(self.spatial_layers, self.spatial_norms):
                    h = h + gat(h, adj_sparse)
                    h = norm(h)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)  # [B, T, N, H]

        return x

    def _sparsify_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Keep only top-k neighbors per node."""
        N = adj.shape[0]
        k = min(self.k_neighbors, N - 1)

        # Get top-k values and indices per row
        values, indices = torch.topk(adj, k, dim=-1)

        # Create sparse adjacency (vectorized)
        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(1, indices, values)

        # Normalize
        row_sum = sparse_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        sparse_adj = sparse_adj / row_sum

        return sparse_adj

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'SpatioTemporalEncoder',
            'k_neighbors': self.k_neighbors,
        })
        return config


class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features [B, N, H]
            adj: Adjacency matrix [N, N]

        Returns:
            output: Updated features [B, N, H]
        """
        B, N, _ = h.shape

        # Linear transformation
        Wh = self.W(h)  # [B, N, H]
        Wh = Wh.view(B, N, self.num_heads, self.head_dim)  # [B, N, heads, head_dim]

        # Compute attention scores
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, N, N, heads, head_dim]
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, heads, head_dim]
        concat = torch.cat([Wh_i, Wh_j], dim=-1)  # [B, N, N, heads, 2*head_dim]

        e = (concat * self.a).sum(dim=-1)  # [B, N, N, heads]
        e = self.leaky_relu(e)

        # Mask with adjacency
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        e = e.masked_fill(mask, float('-inf'))

        # Softmax attention
        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)

        # Apply attention
        Wh_permuted = Wh.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        attention_permuted = attention.permute(0, 3, 1, 2)  # [B, heads, N, N]
        out = torch.matmul(attention_permuted, Wh_permuted)  # [B, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, H]

        return out


# =============================================================================
# Masked Autoencoder Wrapper (for STMAE)
# =============================================================================

@register_encoder('MaskedAutoEncoder')
class MaskedAutoEncoderWrapper(BaseRepresentationEncoder):
    """
    Wrapper for STMAE to conform to BaseRepresentationEncoder interface.

    Uses STMAE's encode() method without masking during inference.
    """

    def __init__(
        self,
        stmae_model: nn.Module = None,
        input_dim: int = 3,
        d_model: int = 64,
        **kwargs
    ):
        super().__init__(input_dim, d_model, **kwargs)

        if stmae_model is not None:
            self.stmae = stmae_model
        else:
            # Build STMAE from config if not provided
            from baselines.STMAE.arch import STMAE
            self.stmae = STMAE(
                input_dim=input_dim,
                hidden_dim=d_model,
                **kwargs
            )

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, N, D]

        Returns:
            embedding: [B, T, N, H]
        """
        # Use STMAE's encode without masking
        embedding, _, _, _ = self.stmae.encode(x, mask_s=0, mask_f=0, **kwargs)
        return embedding

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['type'] = 'MaskedAutoEncoder'
        return config


# =============================================================================
# Helper Modules
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
