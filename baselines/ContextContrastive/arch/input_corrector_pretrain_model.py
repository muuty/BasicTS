"""
Pre-training model for InputSpilloverCorrector.

Same noise injection + denoising objective as other pretrain models,
but uses InputSpilloverCorrector (reliability-gated cross-attention).
"""

import torch
import torch.nn as nn
import numpy as np

from .input_spillover_corrector import InputSpilloverCorrector


class InputCorrectorPretrainModel(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int = 5,
        output_dim: int = 1,
        # Encoder params
        d_model: int = 5,
        hidden_dim: int = 64,
        n_heads: int = 4,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        physical_channels: list = None,
        residual_connection: bool = True,
        # Noise params
        noise_rate_range: tuple = (0.1, 0.5),
        noise_severity_range: tuple = (0.1, 0.5),
        noise_types: list = None,
        clean_prob: float = 0.0,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.d_model = d_model
        self.physical_channels = physical_channels or [0, 1, 2]
        self.noise_rate_range = noise_rate_range
        self.noise_severity_range = noise_severity_range
        self.noise_types = noise_types or ['gaussian', 'drift', 'dead', 'spike']
        self.clean_prob = clean_prob

        self.encoder = InputSpilloverCorrector(
            input_dim=input_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            temporal_layers=temporal_layers,
            dropout=dropout,
            physical_channels=physical_channels,
            residual_connection=residual_connection,
        )

        # Dummy predictor for basicts framework compatibility
        self.dummy_predictor = nn.Linear(input_dim, output_len * output_dim)

    def _inject_noise(self, x: torch.Tensor):
        B, T, N, C = x.shape
        device = x.device

        rate = np.random.uniform(*self.noise_rate_range)
        severity = np.random.uniform(*self.noise_severity_range)

        n_corrupt = max(1, int(N * rate))
        corrupt_idx = torch.randperm(N, device=device)[:n_corrupt]
        noise_mask = torch.zeros(N, dtype=torch.bool, device=device)
        noise_mask[corrupt_idx] = True

        x_noisy = x.clone()
        noise_type = np.random.choice(self.noise_types)

        if noise_type == 'gaussian':
            for ch in self.physical_channels:
                ch_std = x[:, :, :, ch].std().clamp(min=1e-6)
                noise = torch.randn(B, T, n_corrupt, device=device) * (severity * ch_std)
                x_noisy[:, :, corrupt_idx, ch] += noise
                x_noisy[:, :, corrupt_idx, ch].clamp_(min=0)

        elif noise_type == 'bias':
            factors = torch.ones(n_corrupt, device=device)
            under = torch.rand(n_corrupt, device=device) < 0.5
            factors[under] = 1.0 - severity
            factors[~under] = 1.0 + severity
            for ch in self.physical_channels:
                x_noisy[:, :, corrupt_idx, ch] *= factors.unsqueeze(0).unsqueeze(0)
                x_noisy[:, :, corrupt_idx, ch].clamp_(min=0)

        elif noise_type == 'drift':
            directions = torch.ones(n_corrupt, device=device)
            down = torch.rand(n_corrupt, device=device) < 0.5
            directions[down] = -1.0
            t_factors = torch.linspace(0, 1, T, device=device).view(1, T, 1)
            drift = directions.view(1, 1, n_corrupt) * severity * t_factors
            multiplier = 1.0 + drift
            for ch in self.physical_channels:
                x_noisy[:, :, corrupt_idx, ch] *= multiplier
                x_noisy[:, :, corrupt_idx, ch].clamp_(min=0)

        elif noise_type == 'dead':
            for ch in self.physical_channels:
                x_noisy[:, :, corrupt_idx, ch] = 0.0

        elif noise_type == 'stuck':
            for ch in self.physical_channels:
                frozen = x[:, 0:1, corrupt_idx, ch]
                x_noisy[:, :, corrupt_idx, ch] = frozen.expand(B, T, n_corrupt)

        elif noise_type == 'spike':
            spike_mask = torch.rand(B, T, n_corrupt, device=device) < 0.2
            for ch in self.physical_channels:
                ch_std = x[:, :, :, ch].std().clamp(min=1e-6)
                signs = torch.sign(torch.randn(B, T, n_corrupt, device=device))
                spikes = spike_mask * signs * severity * ch_std
                x_noisy[:, :, corrupt_idx, ch] += spikes
                x_noisy[:, :, corrupt_idx, ch].clamp_(min=0)

        return x_noisy, noise_mask

    def forward(self, history_data, future_data=None, batch_seen=None, epoch=None, train=True, **kwargs):
        B, T, N, C = history_data.shape

        if train and np.random.random() >= self.clean_prob:
            x_noisy, noise_mask = self._inject_noise(history_data)
        else:
            x_noisy = history_data
            noise_mask = torch.zeros(N, dtype=torch.bool, device=history_data.device)

        z = self.encoder.encode(x_noisy)

        recon = z[..., self.physical_channels]
        recon_target = history_data[..., self.physical_channels]

        dummy = self.dummy_predictor(z[:, -1, :, :])
        dummy = dummy.reshape(B, N, self.output_len, self.output_dim)
        dummy = dummy.permute(0, 2, 1, 3)

        return {
            'prediction': dummy,
            'recon_pred': recon,
            'recon_target': recon_target,
            'noise_mask': noise_mask,
        }
