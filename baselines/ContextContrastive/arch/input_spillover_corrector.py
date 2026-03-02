"""
Input-Level Spillover Corrector Encoder.

Applies SpilloverCorrector's 3-stage mechanistic structure at the input level,
where noise patterns are directly observable (before spatial attention mixes signals).

Architecture:
  1. Temporal linear attention (per-node, across timesteps) for contextual features
  2. Mean pool over time → per-node summary
  3. Stage 1: Reliability MLP → r in [0,1] per node
  4. Stage 2: Reliability-gated cross-attention (anomaly propagation)
  5. Stage 3: Correction MLP → delta per node (zero-init, broadcast over time)
  6. Output: input + delta (replace strategy, d_model=5)

Key differentiators from previous LinearAttentionDenoisingEncoder:
- NO spatial self-attention (which propagated noise equally to all nodes)
- Reliability-GATED cross-attention: only anomalous nodes contribute to correction
- Interpretable intermediates: reliability r, attention weights, correction delta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder


class LinearAttention(nn.Module):
    """Linear attention with ELU+1 feature map.

    Avoids softmax's exponential amplification of outlier values,
    making it more robust to noisy timesteps.
    Complexity: O(L * d^2) instead of O(L^2 * d).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        KV = torch.einsum('bhld,bhlv->bhdv', K, V)
        num = torch.einsum('bhld,bhdv->bhlv', Q, KV)
        K_sum = K.sum(dim=2)
        denom = torch.einsum('bhld,bhd->bhl', Q, K_sum).unsqueeze(-1).clamp(min=1e-6)

        out = num / denom
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return self.dropout(out)


class TemporalBlock(nn.Module):
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


@register_encoder('InputSpilloverCorrector')
class InputSpilloverCorrector(BaseRepresentationEncoder):
    """
    Input-level denoising encoder with reliability-gated cross-attention.

    3-stage mechanistic structure at input level:
      Stage 1: Reliability estimation r in [0,1] per node
      Stage 2: Reliability-gated cross-attention (anomaly propagation)
      Stage 3: Correction delta per node (zero-init)

    No adjacency matrix required. Replace strategy: d_model = input_dim.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 64,
        n_heads: int = 4,
        temporal_layers: int = 2,
        spatial_layers: int = 0,
        dropout: float = 0.1,
        physical_channels: List[int] = None,
        residual_connection: bool = True,
        time_varying_correction: bool = False,
        correction_mode: str = None,
        disable_reliability: bool = False,
        disable_cross_attn: bool = False,
        gating_mode: str = 'anomaly',
        **kwargs,
    ):
        kwargs.pop('adj_path', None)
        super().__init__(input_dim, d_model, adj_path=None, **kwargs)

        self.hidden_dim = hidden_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        self.residual_connection = residual_connection
        self.disable_reliability = disable_reliability
        self.disable_cross_attn = disable_cross_attn
        self.gating_mode = gating_mode

        # correction_mode supersedes time_varying_correction
        if correction_mode is not None:
            self.correction_mode = correction_mode
        elif time_varying_correction:
            self.correction_mode = 'full'
        else:
            self.correction_mode = 'node'
        n_physical = len(self.physical_channels)

        # Input projection: physical channels → hidden
        self.input_proj = nn.Linear(n_physical, hidden_dim)

        # Temporal self-attention blocks (per node, across timesteps)
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, n_heads, dropout)
            for _ in range(temporal_layers)
        ])

        # V3 path: spatial blocks (reuse TemporalBlock over N dimension)
        self.spatial_blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, n_heads, dropout)
            for _ in range(spatial_layers)
        ])

        # Stage 1: Reliability estimation
        self.reliability = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Stage 2: Reliability-gated cross-attention (V1 path only)
        if spatial_layers == 0:
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)

        # Stage 3: Correction (zero-init for safe training start)
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_physical),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def _compute_reliability_and_propagation(self, h_input, **kwargs):
        """Shared logic for reliability estimation + cross-attention propagation.

        Args:
            h_input: [B_eff, N, hidden] — either pooled or per-timestep
        Returns:
            propagated: [B_eff, N, hidden], r: [B_eff, N, 1]
        """
        need_weights = kwargs.get('return_intermediates', False)

        # Stage 1: Reliability
        if self.disable_reliability:
            r = torch.ones(h_input.shape[0], h_input.shape[1], 1, device=h_input.device)
        else:
            r = self.reliability(h_input)                     # [B_eff, N, 1]

        # Stage 2: Cross-attention
        if self.disable_cross_attn:
            propagated = self.cross_norm(h_input)
        else:
            if self.gating_mode == 'clean':
                gated_signal = r * h_input
            else:  # 'anomaly' (default)
                gated_signal = (1 - r) * h_input
            attn_out, _ = self.cross_attn(
                query=h_input, key=gated_signal, value=gated_signal,
                need_weights=need_weights, average_attn_weights=True,
            )
            propagated = self.cross_norm(h_input + attn_out)  # [B_eff, N, hidden]

        return propagated, r

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, N, D = x.shape

        # Extract physical channels
        x_physical = x[..., self.physical_channels]  # [B, T, N, n_phys]

        # Project to hidden dim
        h = self.input_proj(x_physical)  # [B, T, N, hidden]

        # Temporal self-attention per node
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        for block in self.temporal_blocks:
            h = block(h)
        h = h.reshape(B, N, T, self.hidden_dim)  # [B, N, T, hidden]

        if len(self.spatial_blocks) > 0:
            # V3 path: per-timestep spatial attention → r-gated correction
            h_flat = h.permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_dim)
            for block in self.spatial_blocks:
                h_flat = block(h_flat)              # spatial attn over N
            delta = self.correction(h_flat)         # [B*T, N, n_phys]
            if self.disable_reliability:
                r = torch.ones(B * T, N, 1, device=x.device)
            else:
                r = self.reliability(h_flat)        # [B*T, N, 1]
                delta = delta * (1 - r)             # r-gated: clean(r≈1)→0, noisy(r≈0)→full
            delta = delta.reshape(B, T, N, -1)
            r = r.reshape(B, T, N, 1)

        elif self.correction_mode == 'full':
            # V1 full: all ops per-timestep with cross-attention
            h_flat = h.permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_dim)
            propagated, r = self._compute_reliability_and_propagation(h_flat, **kwargs)
            delta = self.correction(propagated)
            delta = delta.reshape(B, T, N, -1)
            r = r.reshape(B, T, N, 1)

        elif self.correction_mode == 'hybrid':
            # Reliability + cross-attn at node level, correction per-timestep
            h_pooled = h.mean(dim=2)                          # [B, N, hidden]
            _, r = self._compute_reliability_and_propagation(h_pooled, **kwargs)

            # Per-timestep correction from temporal features
            h_flat = h.permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_dim)
            delta = self.correction(h_flat)                   # [B*T, N, n_phys]
            delta = delta.reshape(B, T, N, -1)                # [B, T, N, n_phys]

        else:  # 'node'
            # All pooled, broadcast correction
            h_pooled = h.mean(dim=2)                          # [B, N, hidden]
            propagated, r = self._compute_reliability_and_propagation(h_pooled, **kwargs)
            delta = self.correction(propagated)               # [B, N, n_phys]
            delta = delta.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, N, n_phys]

        # Apply correction
        if self.residual_connection:
            corrected = x_physical + delta
        else:
            corrected = delta

        # Reassemble: replace physical channels, keep tod/dow unchanged
        out = x.clone()
        for i, ch in enumerate(self.physical_channels):
            out[..., ch] = corrected[..., i]

        if kwargs.get('return_intermediates', False):
            return out, {
                'reliability': r,
                'delta': delta,
            }
        return out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'InputSpilloverCorrector',
            'hidden_dim': self.hidden_dim,
            'temporal_layers': len(self.temporal_blocks),
            'spatial_layers': len(self.spatial_blocks),
            'physical_channels': self.physical_channels,
            'residual_connection': self.residual_connection,
            'correction_mode': self.correction_mode,
            'disable_reliability': self.disable_reliability,
            'disable_cross_attn': self.disable_cross_attn,
            'gating_mode': self.gating_mode,
        })
        return config
