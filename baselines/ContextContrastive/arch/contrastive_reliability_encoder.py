"""
Contrastive Reliability Encoder (Adjacency-Free).

Denoising encoder that computes reliability from cross-prediction consistency
using learned attention, without any pre-defined adjacency matrix.

Architecture:
  Stage 1: Per-node temporal denoiser (dilated causal convolutions)
           - Same as DenoisingEncoder: captures temporal patterns per node
           - No cross-node interaction (prevents noise spillover)

  Stage 2: Cross-prediction attention (exclude-self)
           - Each node attends to ALL other nodes (self excluded via mask)
           - Learned attention replaces pre-defined adjacency matrix
           - Computes reliability = similarity(h_node, h_context)
           - h_context: what neighbors predict this node should look like

  Stage 3: Gated spatial fusion
           - Combines per-node temporal features with neighbor context
           - Gated residual: h' = h + gate * h_context
           - Replaces graph convolution (no adjacency needed)

Reliability mechanism:
  - Clean/normal nodes: temporal pattern matches what neighbors predict → high reliability
  - Stuck sensors: constant temporal → doesn't match neighbors' expectation → low reliability
  - Gaussian noise: distorted temporal → partial mismatch → moderate reliability

Key advantage over reconstruction-error-based reliability:
  - Stuck noise has LOW reconstruction error (values look normal) → old approach gives HIGH reliability (wrong)
  - Cross-prediction catches stuck because neighbors show variation while stuck node is constant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder
from .encoders import DilatedConvBlock


class ExcludeSelfAttention(nn.Module):
    """Multi-head attention where each node attends to all OTHER nodes.

    Self-exclusion via masking the diagonal of attention scores to -inf.
    This prevents information leakage: a node's reliability should be
    judged purely by comparing with OTHER nodes' patterns.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, D] node representations (temporally pooled)
        Returns:
            h_context: [B, N, D] aggregated neighbor context (self excluded)
        """
        B, N, D = h.shape
        H = self.num_heads
        d = self.head_dim

        Q = self.W_Q(h).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        K = self.W_K(h).view(B, N, H, d).transpose(1, 2)
        V = self.W_V(h).view(B, N, H, d).transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / (d ** 0.5)  # [B, H, N, N]

        # Exclude self: mask diagonal to -inf before softmax
        self_mask = torch.eye(N, device=h.device, dtype=torch.bool)  # [N, N]
        scores = scores.masked_fill(self_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # [B, H, N, N]
        attn = self.dropout(attn)

        out = (attn @ V)  # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        return self.out_proj(out)


@register_encoder('ContrastiveReliabilityEncoder')
class ContrastiveReliabilityEncoder(BaseRepresentationEncoder):
    """
    Denoising encoder with cross-prediction contrastive reliability.
    Fully adjacency-free: all spatial interactions learned via attention.

    The reliability score captures whether a node's temporal pattern is
    consistent with what its learned neighbors predict, using cosine
    similarity between the node's own representation and the cross-predicted
    context from other nodes.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 32,
        temporal_layers: int = 4,
        spatial_layers: int = 1,  # unused, kept for config compat
        num_heads: int = 4,
        dropout: float = 0.1,
        physical_channels: List[int] = None,
        reliability_scale_init: float = 5.0,
        **kwargs,
    ):
        # No adj_path needed - fully adjacency-free
        kwargs.pop('adj_path', None)
        kwargs.pop('k_neighbors', None)
        super().__init__(input_dim, d_model, adj_path=None, **kwargs)

        self.hidden_dim = hidden_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        n_physical = len(self.physical_channels)

        # Stage 1: Temporal denoiser (per-node, same as DenoisingEncoder)
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

        # Stage 2: Cross-prediction attention (exclude-self)
        self.cross_pred_attn = ExcludeSelfAttention(hidden_dim, num_heads, dropout)
        self.cross_pred_norm = nn.LayerNorm(hidden_dim)

        # Reliability: learned scale and bias for sigmoid(scale * cos_sim + bias)
        self.reliability_scale = nn.Parameter(torch.tensor(reliability_scale_init))
        self.reliability_bias = nn.Parameter(torch.tensor(0.0))

        # Stage 3: Gated spatial fusion (replaces graph conv)
        # Gate decides how much neighbor context to incorporate per dimension
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden -> denoised physical channels
        self.output_proj = nn.Linear(hidden_dim, n_physical)

    def encode(self, x: torch.Tensor, return_reliability: bool = False, **kwargs):
        """
        Args:
            x: [B, T, N, D_in] where D_in includes physical + tod + dow
            return_reliability: if True, return (denoised, reliability) tuple
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

        # --- Stage 2: Cross-prediction attention ---
        # Pool temporal dimension for cross-node comparison
        h_node = h.mean(dim=1)  # [B, N, hidden]

        # Cross-attention: each node queries all OTHER nodes
        h_context = self.cross_pred_attn(h_node)  # [B, N, hidden]
        h_context = self.cross_pred_norm(h_context)

        # Compute reliability from cross-prediction consistency
        cos_sim = F.cosine_similarity(h_node, h_context, dim=-1)  # [B, N]
        reliability = torch.sigmoid(
            self.reliability_scale * cos_sim + self.reliability_bias
        )  # [B, N]
        # Expand to per-timestep: [B, T, N, 1]
        reliability = reliability.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1)

        # --- Stage 3: Gated spatial fusion ---
        # Broadcast h_context to all timesteps and fuse with temporal features
        h_context_expanded = h_context.unsqueeze(1).expand_as(h)  # [B, T, N, hidden]
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([h, h_context_expanded], dim=-1))
        )  # [B, T, N, hidden]
        h = self.fusion_norm(h + gate * h_context_expanded)

        # Project back to physical channel space
        denoised = self.output_proj(h)  # [B, T, N, n_physical]

        # Reassemble: replace physical channels with denoised, keep tod/dow unchanged
        out = x.clone()
        for i, ch in enumerate(self.physical_channels):
            out[..., ch] = denoised[..., i]

        if return_reliability:
            return out, reliability.contiguous()
        return out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'ContrastiveReliabilityEncoder',
            'hidden_dim': self.hidden_dim,
            'temporal_layers': len(self.temporal_blocks),
            'num_heads': self.cross_pred_attn.num_heads,
            'physical_channels': self.physical_channels,
        })
        return config
