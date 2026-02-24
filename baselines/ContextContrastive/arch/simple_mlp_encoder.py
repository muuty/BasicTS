"""
Simple MLP Encoder for Ablation Study (A2).

Per-node, per-timestep MLP with residual connection.
No temporal processing, no spatial processing, no channel mixer.
Tests whether the spatio-temporal components are necessary.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .base_encoder import BaseRepresentationEncoder, register_encoder


@register_encoder('SimpleMLPEncoder')
class SimpleMLPEncoder(BaseRepresentationEncoder):
    """Per-node, per-timestep MLP encoder. No temporal or spatial processing."""

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 5,
        hidden_dim: int = 64,
        physical_channels: List[int] = None,
        residual_connection: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ):
        kwargs.pop('adj_path', None)
        super().__init__(input_dim, d_model, **kwargs)

        self.hidden_dim = hidden_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        self.residual_connection = residual_connection
        n_physical = len(self.physical_channels)

        self.mlp = nn.Sequential(
            nn.Linear(n_physical, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_physical),
        )

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D_in]
        Returns:
            [B, T, N, D_in] with denoised physical channels
        """
        x_physical = x[..., self.physical_channels]
        correction = self.mlp(x_physical)

        if self.residual_connection:
            denoised = x_physical + correction
        else:
            denoised = correction

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
            'type': 'SimpleMLPEncoder',
            'hidden_dim': self.hidden_dim,
            'physical_channels': self.physical_channels,
            'residual_connection': self.residual_connection,
        })
        return config


class SimpleMLPPretrainModel(nn.Module):
    """Pre-training wrapper for SimpleMLPEncoder. Reuses noise injection from DenoisingPretrainModel."""

    def __init__(self, **kwargs):
        super().__init__()
        from .denoising_pretrain_model import DenoisingPretrainModel
        # Create a DenoisingPretrainModel for noise injection logic, then replace encoder
        self._base = DenoisingPretrainModel(**kwargs)
        self._base.encoder = SimpleMLPEncoder(
            input_dim=kwargs.get('input_dim', 5),
            d_model=kwargs.get('d_model', 5),
            hidden_dim=kwargs.get('hidden_dim', 64),
            physical_channels=kwargs.get('physical_channels', [0, 1, 2]),
            residual_connection=kwargs.get('residual_connection', True),
            dropout=kwargs.get('dropout', 0.1),
        )

    def forward(self, *args, **kwargs):
        return self._base(*args, **kwargs)
