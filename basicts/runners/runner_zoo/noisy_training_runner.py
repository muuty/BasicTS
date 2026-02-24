"""
Noisy Training Runner: Training-time noise augmentation for robustness.

Injects random noise into input data during training only.
The model learns to predict clean targets from noisy inputs,
making it inherently robust to sensor degradation at test time.

Architecture-agnostic: works with any downstream forecasting model.
"""

import random
from typing import Dict, Optional

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class NoisyTrainingRunner(SimpleTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        noise_cfg = cfg.get('NOISE_AUGMENTATION', {})
        self.noise_rate_range = noise_cfg.get('rate_range', [0.05, 0.3])
        self.noise_types = noise_cfg.get('types', ['gaussian', 'bias', 'stuck', 'drift'])
        self.noise_severity_range = noise_cfg.get('severity_range', [0.1, 0.5])
        self.noise_physical_channels = noise_cfg.get('physical_channels', [0, 1, 2])
        self.noise_prob = noise_cfg.get('prob', 0.5)

    def _inject_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Inject random noise into training inputs. [B, T, N, C]"""
        if random.random() > self.noise_prob:
            return data

        B, T, N, C = data.shape
        corrupted = data.clone()

        noise_type = random.choice(self.noise_types)
        severity = random.uniform(*self.noise_severity_range)
        rate = random.uniform(*self.noise_rate_range)
        n_corrupt = max(1, int(N * rate))
        corrupt_idx = torch.randperm(N, device=data.device)[:n_corrupt]

        if noise_type == 'gaussian':
            for ch in self.noise_physical_channels:
                ch_std = data[:, :, :, ch].std().item() + 1e-8
                noise = torch.randn(B, T, n_corrupt, device=data.device) * severity * ch_std
                corrupted[:, :, corrupt_idx, ch] += noise

        elif noise_type == 'bias':
            factors = torch.ones(n_corrupt, device=data.device)
            under = torch.rand(n_corrupt) < 0.5
            factors[under] = 1.0 - severity
            factors[~under] = 1.0 + severity
            factors = factors.view(1, 1, n_corrupt)
            for ch in self.noise_physical_channels:
                corrupted[:, :, corrupt_idx, ch] *= factors

        elif noise_type == 'stuck':
            for ch in self.noise_physical_channels:
                frozen = corrupted[:, 0:1, corrupt_idx, ch]
                corrupted[:, :, corrupt_idx, ch] = frozen.expand(B, T, n_corrupt)

        elif noise_type == 'drift':
            directions = torch.ones(n_corrupt, device=data.device)
            directions[torch.rand(n_corrupt) < 0.5] = -1.0
            t_ramp = torch.linspace(0, 1, T, device=data.device).view(1, T, 1)
            multiplier = 1.0 + directions.view(1, 1, n_corrupt) * severity * t_ramp
            for ch in self.noise_physical_channels:
                corrupted[:, :, corrupt_idx, ch] *= multiplier

        return corrupted

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None,
                train: bool = True, **kwargs) -> Dict:

        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Noise augmentation: training only, before feature selection
        if train:
            history_data = self._inject_noise(history_data)

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)
        return model_return
