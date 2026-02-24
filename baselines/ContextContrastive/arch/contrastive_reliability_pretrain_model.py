"""
Contrastive Reliability Pre-training Model.

Same noise injection as DenoisingPretrainModel, but uses ContrastiveReliabilityEncoder
which computes reliability from cross-prediction consistency (adjacency-free).

The reliability score is a natural byproduct of the cross-prediction mechanism:
no separate reliability loss needed. Standard denoising loss trains everything end-to-end.
"""

import torch
import torch.nn as nn
import numpy as np

from .contrastive_reliability_encoder import ContrastiveReliabilityEncoder


class ContrastiveReliabilityPretrainModel(nn.Module):
    """
    Pre-training wrapper for ContrastiveReliabilityEncoder.

    Forward pass:
    1. Inject random noise into a fraction of nodes (physical channels only)
    2. Encode via ContrastiveReliabilityEncoder (denoises + computes reliability)
    3. Return denoised output, clean target, noise mask, and reliability
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int = 5,
        output_dim: int = 1,
        # Encoder params
        d_model: int = 5,
        hidden_dim: int = 32,
        temporal_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        reliability_scale_init: float = 5.0,
        # Noise params
        noise_rate_range: tuple = (0.1, 0.5),
        noise_severity_range: tuple = (0.1, 0.5),
        physical_channels: list = None,
        **kwargs,
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

        # Contrastive reliability encoder (adjacency-free)
        self.encoder = ContrastiveReliabilityEncoder(
            input_dim=input_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            num_heads=num_heads,
            dropout=dropout,
            physical_channels=physical_channels,
            reliability_scale_init=reliability_scale_init,
        )

        # Dummy predictor for basicts framework compatibility
        self.dummy_predictor = nn.Linear(input_dim, output_len * output_dim)

    def _inject_noise(self, x: torch.Tensor):
        """Inject random noise into physical channels.

        Returns:
            x_noisy: corrupted input
            noise_mask: [N] boolean, True = corrupted node
        """
        B, T, N, C = x.shape
        device = x.device

        rate = np.random.uniform(*self.noise_rate_range)
        severity = np.random.uniform(*self.noise_severity_range)

        n_corrupt = max(1, int(N * rate))
        corrupt_idx = torch.randperm(N, device=device)[:n_corrupt]
        noise_mask = torch.zeros(N, dtype=torch.bool, device=device)
        noise_mask[corrupt_idx] = True

        x_noisy = x.clone()

        noise_type = np.random.choice(['gaussian', 'bias', 'stuck', 'drift'])

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

        elif noise_type == 'stuck':
            for ch in self.physical_channels:
                frozen_val = x_noisy[:, 0:1, corrupt_idx, ch]
                x_noisy[:, :, corrupt_idx, ch] = frozen_val.expand(B, T, n_corrupt)

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

        return x_noisy, noise_mask

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs,
    ) -> dict:
        B, T, N, C = history_data.shape

        if train:
            x_noisy, noise_mask = self._inject_noise(history_data)
        else:
            x_noisy = history_data
            noise_mask = torch.zeros(N, dtype=torch.bool, device=history_data.device)

        # Encode (denoises + computes reliability)
        z, reliability = self.encoder.encode(x_noisy, return_reliability=True)

        # Extract physical channels as reconstruction
        recon = z[..., self.physical_channels]  # [B, T, N, n_physical]
        recon_target = history_data[..., self.physical_channels]

        # Dummy prediction for framework compatibility
        dummy = self.dummy_predictor(z[:, -1, :, :])
        dummy = dummy.reshape(B, N, self.output_len, self.output_dim)
        dummy = dummy.permute(0, 2, 1, 3)

        return {
            'prediction': dummy,
            'recon_pred': recon,
            'recon_target': recon_target,
            'noise_mask': noise_mask,
            'reliability': reliability,  # [B, T, N, 1]
        }
