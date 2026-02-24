"""
Noise-Invariant Pre-training Model.

Extends DenoisingPretrainModel with alignment loss:
  - L_recon: MAE(encoder(x_noisy), x_clean) — denoising objective
  - L_align: MSE(encoder(x_noisy), encoder(x_clean)) — noise-invariance
  - L_pass:  MAE on clean nodes — prevent over-denoising

The alignment term forces the encoder to produce the SAME representation
regardless of whether input is clean or noisy.
"""

import torch
import torch.nn as nn
import numpy as np

from .denoising_encoder import DenoisingEncoder


class NoiseInvariantPretrainModel(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int = 5,
        output_dim: int = 1,
        d_model: int = 5,
        hidden_dim: int = 64,
        temporal_layers: int = 4,
        spatial_layers: int = 1,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        adj_path: str = None,
        noise_rate_range: tuple = (0.1, 0.5),
        noise_severity_range: tuple = (0.1, 0.5),
        physical_channels: list = None,
        residual_connection: bool = True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.physical_channels = physical_channels or [0, 1, 2]
        self.noise_rate_range = noise_rate_range
        self.noise_severity_range = noise_severity_range

        self.encoder = DenoisingEncoder(
            input_dim=input_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            spatial_layers=spatial_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            adj_path=adj_path,
            physical_channels=physical_channels,
            residual_connection=residual_connection,
        )

        # Dummy predictor for basicts framework compatibility
        self.dummy_predictor = nn.Linear(input_dim, output_len * output_dim)

    def _inject_noise(self, x: torch.Tensor):
        """Inject random noise. Returns (x_noisy, noise_mask[N])."""
        B, T, N, C = x.shape
        device = x.device

        rate = np.random.uniform(*self.noise_rate_range)
        severity = np.random.uniform(*self.noise_severity_range)
        n_corrupt = max(1, int(N * rate))
        corrupt_idx = torch.randperm(N, device=device)[:n_corrupt]
        noise_mask = torch.zeros(N, dtype=torch.bool, device=device)
        noise_mask[corrupt_idx] = True

        x_noisy = x.clone()
        noise_type = np.random.choice(['gaussian', 'bias', 'drift'])

        if noise_type == 'gaussian':
            for ch in self.physical_channels:
                ch_std = x[:, :, :, ch].std().clamp(min=1e-6)
                noise = torch.randn(B, T, n_corrupt, device=device) * (severity * ch_std)
                x_noisy[:, :, corrupt_idx, ch] += noise

        elif noise_type == 'bias':
            factors = torch.ones(n_corrupt, device=device)
            under = torch.rand(n_corrupt, device=device) < 0.5
            factors[under] = 1.0 - severity
            factors[~under] = 1.0 + severity
            for ch in self.physical_channels:
                x_noisy[:, :, corrupt_idx, ch] *= factors.unsqueeze(0).unsqueeze(0)

        elif noise_type == 'drift':
            directions = torch.ones(n_corrupt, device=device)
            directions[torch.rand(n_corrupt, device=device) < 0.5] = -1.0
            t_factors = torch.linspace(0, 1, T, device=device).view(1, T, 1)
            multiplier = 1.0 + directions.view(1, 1, n_corrupt) * severity * t_factors
            for ch in self.physical_channels:
                x_noisy[:, :, corrupt_idx, ch] *= multiplier

        return x_noisy, noise_mask

    def forward(self, history_data, future_data=None, batch_seen=None,
                epoch=None, train=True, **kwargs):
        B, T, N, C = history_data.shape

        if train:
            x_noisy, noise_mask = self._inject_noise(history_data)

            # Encode both clean and noisy
            z_clean = self.encoder.encode(history_data)
            z_noisy = self.encoder.encode(x_noisy)

            recon_noisy = z_noisy[..., self.physical_channels]
            recon_clean = z_clean[..., self.physical_channels]
            recon_target = history_data[..., self.physical_channels]
        else:
            z_clean = self.encoder.encode(history_data)
            noise_mask = torch.zeros(N, dtype=torch.bool, device=history_data.device)
            recon_noisy = z_clean[..., self.physical_channels]
            recon_clean = recon_noisy
            recon_target = history_data[..., self.physical_channels]

        # Dummy prediction for framework compatibility
        dummy = self.dummy_predictor(z_clean[:, -1, :, :])
        dummy = dummy.reshape(B, N, self.output_len, self.output_dim).permute(0, 2, 1, 3)

        return {
            'prediction': dummy,
            'recon_noisy': recon_noisy,     # encoder(x_noisy) physical channels
            'recon_clean': recon_clean,     # encoder(x_clean) physical channels
            'recon_target': recon_target,   # clean x physical channels
            'noise_mask': noise_mask,
        }
