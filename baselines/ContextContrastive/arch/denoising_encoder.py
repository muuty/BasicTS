"""
Denoising Encoder for Noise-Robust Traffic Prediction.

Two-stage architecture:
  Stage 1: Per-node temporal denoiser (dilated causal convolutions)
           - Captures temporal continuity violations
           - Detects cross-channel inconsistencies (flow vs occ vs speed)
           - No cross-node interaction (prevents noise spillover)

  Stage 2: Spatial consistency checker (graph convolution, NOT attention)
           - Local neighbor comparison to detect spatial outliers
           - Graph conv (like STGCN) instead of attention (prevents amplification)
           - Residual connection to Stage 1

Replace strategy: D_out = D_in, so downstream model needs NO changes.
The encoder acts as a transparent filter: clean data passes through, noisy data gets corrected.

Pre-training: denoising autoencoder with synthetic noise injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder
from .encoders import DilatedConvBlock


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer (message-passing, NOT attention).

    h_out = h + dropout(W2 * ReLU(W1 * (adj @ h)))

    Uses local graph structure only. No attention = no noise amplification.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, D]
            adj: [N, N] (row-normalized)
        Returns:
            [B, N, D]
        """
        # Message passing: aggregate neighbor features
        agg = torch.matmul(adj, h)  # [B, N, D]
        agg = self.fc1(agg)
        agg = F.relu(agg)
        agg = self.fc2(agg)
        agg = self.dropout(agg)
        return h + agg  # residual


@register_encoder('DenoisingEncoder')
class DenoisingEncoder(BaseRepresentationEncoder):
    """
    Two-stage denoising encoder for noise-robust traffic prediction.

    Stage 1: Temporal dilated conv (per-node, no cross-node interaction)
    Stage 2: Spatial graph conv (local neighbors, not global attention)

    Only physical channels are denoised; tod/dow pass through unchanged.
    Output dim = input dim (replace strategy).
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 32,
        temporal_layers: int = 4,
        spatial_layers: int = 1,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        adj_path: str = None,
        physical_channels: List[int] = None,
        reliability_head: bool = False,
        residual_connection: bool = False,
        **kwargs,
    ):
        super().__init__(input_dim, d_model, adj_path=adj_path, **kwargs)

        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.physical_channels = physical_channels or [0, 1, 2]
        self.has_reliability_head = reliability_head
        self.residual_connection = residual_connection
        n_physical = len(self.physical_channels)

        # Stage 1: Temporal denoiser (operates on physical channels only)
        self.input_proj = nn.Linear(n_physical, hidden_dim)

        self.temporal_blocks = nn.ModuleList([
            DilatedConvBlock(
                hidden_dim, hidden_dim,
                kernel_size=3, dilation=2**i, dropout=dropout,
            )
            for i in range(temporal_layers)
        ])

        # Cross-channel consistency mixer
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)

        # Stage 2: Spatial consistency (graph conv)
        self.spatial_convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(spatial_layers)
        ])
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(spatial_layers)
        ])

        # Output projection: hidden -> denoised physical channels
        self.output_proj = nn.Linear(hidden_dim, n_physical)

        # Reliability head: estimates per-node, per-timestep reliability (0-1)
        if reliability_head:
            self.reliability_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Cache for sparsified adjacency
        self._adj_sparse = None

    def _sparsify_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Keep only top-k neighbors per node, row-normalized."""
        N = adj.shape[0]
        k = min(self.k_neighbors, N - 1)
        values, indices = torch.topk(adj, k, dim=-1)
        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(1, indices, values)
        row_sum = sparse_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return sparse_adj / row_sum

    def _get_adj_sparse(self) -> torch.Tensor:
        """Get cached sparsified adjacency."""
        if self._adj_sparse is None and self.adj is not None:
            self._adj_sparse = self._sparsify_adj(self.adj)
        return self._adj_sparse

    def encode(self, x: torch.Tensor, return_reliability: bool = False, **kwargs):
        """
        Args:
            x: [B, T, N, D_in] where D_in includes physical + tod + dow
            return_reliability: if True and reliability_head enabled, return
                (denoised, reliability) tuple instead of just denoised
        Returns:
            If return_reliability=False: cleaned [B, T, N, D_in]
            If return_reliability=True:  (cleaned [B, T, N, D_in], reliability [B, T, N, 1])
        """
        B, T, N, D = x.shape

        # Extract physical channels only for denoising
        x_physical = x[..., self.physical_channels]  # [B, T, N, n_physical]

        # --- Stage 1: Temporal denoising (per-node) ---
        h = self.input_proj(x_physical)  # [B, T, N, hidden]

        # Reshape for Conv1d: [B*N, hidden, T]
        h = h.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, T)

        for block in self.temporal_blocks:
            h = block(h)  # [B*N, hidden, T]

        # Back to [B, T, N, hidden]
        h = h.reshape(B, N, self.hidden_dim, T).permute(0, 3, 1, 2)

        # Cross-channel mixing
        residual = h
        h = self.channel_mixer(h)
        h = self.channel_norm(residual + h)

        # --- Stage 2: Spatial consistency ---
        adj_sparse = self._get_adj_sparse()
        if adj_sparse is not None:
            for conv, norm in zip(self.spatial_convs, self.spatial_norms):
                # Process each timestep
                h_out = []
                for t in range(T):
                    ht = conv(h[:, t], adj_sparse)  # [B, N, hidden]
                    h_out.append(ht)
                h = torch.stack(h_out, dim=1)  # [B, T, N, hidden]
                h = norm(h)

        # Compute reliability from hidden representation (before output_proj)
        reliability = None
        if return_reliability and self.has_reliability_head:
            reliability = self.reliability_proj(h)  # [B, T, N, 1]

        # Project back to physical channel space
        correction = self.output_proj(h)  # [B, T, N, n_physical]

        # Apply residual: encoder learns correction term, not full reconstruction
        if self.residual_connection:
            denoised = x_physical + correction
        else:
            denoised = correction

        # Reassemble: replace physical channels with denoised, keep tod/dow unchanged
        out = x.clone()
        for i, ch in enumerate(self.physical_channels):
            out[..., ch] = denoised[..., i]

        if reliability is not None:
            return out, reliability
        return out  # [B, T, N, D_in]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'DenoisingEncoder',
            'hidden_dim': self.hidden_dim,
            'temporal_layers': len(self.temporal_blocks),
            'spatial_layers': len(self.spatial_convs),
            'k_neighbors': self.k_neighbors,
            'physical_channels': self.physical_channels,
            'reliability_head': self.has_reliability_head,
            'residual_connection': self.residual_connection,
        })
        return config
