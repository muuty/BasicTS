"""
Input-Level Spillover Corrector V2: Separate Memory Attention + Gate.

Minimal architecture addressing v1's fundamental flaw (per-node temporal-only reliability).

Architecture:
  x[B,T,N,C] → flatten(T*n_phys) → Linear → h[B,N,d]
  Memory M[K,d] (learnable normal-pattern prototypes)
  h_spatial = SelfAttn(Q=h, KV=h)            ← node-to-node (spatial context)
  h_memory  = CrossAttn(Q=h, KV=M)           ← node-to-memory (normal reference)
  g = sigmoid(Linear(h))                      ← input-dependent gate
  h_ctx = LayerNorm(h + g * h_memory + (1-g) * h_spatial)
  r = sigmoid(MLP(h_ctx))                     ← reliability
  δ = CorrectionMLP(h_ctx) * (1-r)            ← r-gated correction, zero-init
  output = x + δ (broadcast over T)

Key design: memory and nodes use SEPARATE softmax (no dilution).
Gate controls per-node memory influence — anomalous nodes learn to rely more on memory.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder


@register_encoder('InputSpilloverCorrectorV2')
class InputSpilloverCorrectorV2(BaseRepresentationEncoder):

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        physical_channels: List[int] = None,
        residual_connection: bool = True,
        n_memory_slots: int = 32,
        input_len: int = 12,
        **kwargs,
    ):
        kwargs.pop('adj_path', None)
        super().__init__(input_dim, d_model, adj_path=None, **kwargs)

        self.hidden_dim = hidden_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        self.residual_connection = residual_connection
        n_physical = len(self.physical_channels)

        # Flatten temporal + physical → hidden
        self.flatten_proj = nn.Linear(input_len * n_physical, hidden_dim)

        # Learnable memory bank: K normal-pattern prototypes
        self.memory = nn.Parameter(torch.randn(n_memory_slots, hidden_dim) * 0.02)

        # Separate attention streams (no shared softmax)
        self.spatial_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.memory_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.ctx_norm = nn.LayerNorm(hidden_dim)

        # Input-dependent gate: per-node control of memory influence
        self.gate_head = nn.Linear(hidden_dim, 1)

        # Reliability: 2-layer MLP
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Correction: zero-init output for identity start
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_physical),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, N, D = x.shape

        # Extract and flatten physical channels
        x_physical = x[..., self.physical_channels]                     # [B, T, N, n_phys]
        x_flat = x_physical.permute(0, 2, 1, 3).reshape(B, N, -1)      # [B, N, T*n_phys]
        h = self.flatten_proj(x_flat)                                   # [B, N, hidden]

        # Separate attention streams
        need_weights = kwargs.get('return_intermediates', False)

        # 1. Spatial self-attention: node-to-node
        h_spatial, spatial_w = self.spatial_attn(
            query=h, key=h, value=h,
            need_weights=need_weights, average_attn_weights=True,
        )

        # 2. Memory cross-attention: node-to-memory (separate softmax)
        M = self.memory.unsqueeze(0).expand(B, -1, -1)                  # [B, K, hidden]
        h_memory, memory_w = self.memory_attn(
            query=h, key=M, value=M,
            need_weights=need_weights, average_attn_weights=True,
        )

        # 3. Input-dependent gate
        g = torch.sigmoid(self.gate_head(h))                            # [B, N, 1]

        # 4. Combine: gate controls memory vs spatial balance
        h_combined = g * h_memory + (1 - g) * h_spatial
        h_ctx = self.ctx_norm(h + h_combined)                           # [B, N, hidden]

        # Reliability from spatially-contextualized features
        r = torch.sigmoid(self.reliability_head(h_ctx))                 # [B, N, 1]

        # Correction: gated by (1-r), zero-init start
        delta = self.correction(h_ctx) * (1 - r)                       # [B, N, n_phys]
        delta_broadcast = delta.unsqueeze(1).expand(-1, T, -1, -1)      # [B, T, N, n_phys]

        # Apply correction
        if self.residual_connection:
            corrected = x_physical + delta_broadcast
        else:
            corrected = delta_broadcast

        out = x.clone()
        for i, ch in enumerate(self.physical_channels):
            out[..., ch] = corrected[..., i]

        if kwargs.get('return_intermediates', False):
            return out, {
                'reliability': r,           # [B, N, 1]
                'delta': delta,             # [B, N, n_phys]
                'gate': g,                  # [B, N, 1]
                'spatial_attn_w': spatial_w,  # [B, N, N]
                'memory_attn_w': memory_w,    # [B, N, K]
            }
        if kwargs.get('return_reliability', False):
            return out, r                   # [B, N, 1]
        return out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'InputSpilloverCorrectorV2',
            'hidden_dim': self.hidden_dim,
            'n_memory_slots': self.memory.shape[0],
            'physical_channels': self.physical_channels,
            'residual_connection': self.residual_connection,
        })
        return config
