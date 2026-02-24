"""
Reliability-Gated MLP Encoder.

x_hat = x + g(x) * delta(x)

- delta(x): correction MLP (same as SimpleMLPEncoder)
- g(x): reliability gate (sigmoid) — learns WHEN and HOW MUCH to correct

Key insight: clean nodes get g≈0 (no correction), noisy nodes get g≈1 (full correction).
This reduces spillover to clean nodes while maintaining noise robustness.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder


@register_encoder('GatedMLPEncoder')
class GatedMLPEncoder(BaseRepresentationEncoder):
    """Per-node, per-timestep MLP with reliability gate."""

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 64,
        physical_channels: List[int] = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        kwargs.pop('adj_path', None)
        super().__init__(input_dim, d_model, **kwargs)

        self.hidden_dim = hidden_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        n_physical = len(self.physical_channels)

        # Correction pathway: delta(x)
        self.correction_mlp = nn.Sequential(
            nn.Linear(n_physical, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_physical),
        )

        # Gate pathway: g(x) -> [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(n_physical, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_physical),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D_in]
        Returns:
            [B, T, N, D_in] with gated denoised physical channels
        """
        x_physical = x[..., self.physical_channels]  # [B, T, N, C_phys]

        delta = self.correction_mlp(x_physical)       # [B, T, N, C_phys]
        gate = self.gate_mlp(x_physical)               # [B, T, N, C_phys]

        denoised = x_physical + gate * delta

        out = x.clone()
        for i, ch in enumerate(self.physical_channels):
            out[..., ch] = denoised[..., i]
        return out

    def _extract_encoder_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoder_state_dict = {}
        for key, value in state_dict.items():
            for prefix in ['_base.encoder.', 'encoder.']:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    encoder_state_dict[new_key] = value
                    break
        return encoder_state_dict

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'type': 'GatedMLPEncoder',
            'hidden_dim': self.hidden_dim,
            'physical_channels': self.physical_channels,
        })
        return config


class GatedMLPPretrainModel(nn.Module):
    """Pre-training wrapper for GatedMLPEncoder."""

    def __init__(self, **kwargs):
        super().__init__()
        from .denoising_pretrain_model import DenoisingPretrainModel
        self._base = DenoisingPretrainModel(**kwargs)
        self._base.encoder = GatedMLPEncoder(
            input_dim=kwargs.get('input_dim', 5),
            d_model=kwargs.get('d_model', 5),
            hidden_dim=kwargs.get('hidden_dim', 64),
            physical_channels=kwargs.get('physical_channels', [0, 1, 2]),
            dropout=kwargs.get('dropout', 0.1),
        )

    def forward(self, *args, **kwargs):
        return self._base(*args, **kwargs)
