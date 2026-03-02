"""Noise robustness evaluation for traffic forecasting runners.

Test-time noise injection to measure model vulnerability to sensor faults.
Integrated into base_tsf_runner.test() via CFG.EVAL.NOISE_ROBUSTNESS = True.

Noise types (realistic sensor faults):
  - gaussian: Additive Gaussian noise on physical channels
  - bias:     Systematic under/over-counting (multiplicative)
  - stuck:    Sensor frozen at first timestep value
  - drift:    Gradual calibration shift over time window
  - dead:     All physical channels report 0
"""

import os

import numpy as np
import torch
from typing import List, Tuple, Optional

# Standard noise configs: (type, rate, severity, label)
# rate = fraction of functional nodes corrupted
# severity = noise-specific intensity parameter
DEFAULT_NOISE_CONFIGS = [
    ('gaussian', 0.30, 0.3, 'gauss_s03_r30'),
    ('bias', 0.30, 0.3, 'bias_s03_r30'),
    ('stuck', 0.30, 0, 'stuck_r30'),
    ('drift', 0.30, 0.3, 'drift_s03_r30'),
    ('dead', 0.30, 0, 'dead_r30'),
]

SEED = 42


def inject_noise(inputs: torch.Tensor, noise_type: str, corrupt_nodes: np.ndarray,
                 severity: float, physical_channels: List[int],
                 rng: np.random.RandomState) -> torch.Tensor:
    """Inject noise into raw input tensor [B, T, N, C].

    Operates on raw (unnormalized) data. Non-physical channels (tod, dow) are unchanged.
    """
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]
    cn = list(corrupt_nodes)

    if noise_type == 'gaussian':
        for ch in physical_channels:
            ch_std = corrupted[:, :, :, ch].std().item() + 1e-8
            noise = torch.tensor(
                rng.normal(0, severity * ch_std, (B, T, len(cn))),
                dtype=corrupted.dtype, device=corrupted.device)
            corrupted[:, :, cn, ch] += noise
        for ch in physical_channels:
            corrupted[:, :, cn, ch].clamp_(min=0)

    elif noise_type == 'bias':
        n = len(cn)
        factors = np.ones(n)
        under = rng.random(n) < 0.5
        factors[under] = 1.0 - severity
        factors[~under] = 1.0 + severity
        factors_t = torch.tensor(factors, dtype=corrupted.dtype,
                                 device=corrupted.device).view(1, 1, n)
        for ch in physical_channels:
            corrupted[:, :, cn, ch] *= factors_t
        for ch in physical_channels:
            corrupted[:, :, cn, ch].clamp_(min=0)

    elif noise_type == 'stuck':
        for ch in physical_channels:
            frozen = corrupted[:, 0:1, cn, ch]
            corrupted[:, :, cn, ch] = frozen.expand(B, T, len(cn))

    elif noise_type == 'drift':
        n = len(cn)
        directions = np.ones(n)
        directions[rng.random(n) < 0.5] = -1.0
        t_factors = torch.linspace(0, 1, T, device=corrupted.device).view(1, T, 1)
        drift = torch.tensor(directions, dtype=corrupted.dtype,
                             device=corrupted.device).view(1, 1, n)
        multiplier = 1.0 + drift * severity * t_factors
        for ch in physical_channels:
            corrupted[:, :, cn, ch] *= multiplier
        for ch in physical_channels:
            corrupted[:, :, cn, ch].clamp_(min=0)

    elif noise_type == 'dead':
        for ch in physical_channels:
            corrupted[:, :, cn, ch] = 0.0

    return corrupted


def select_corrupt_nodes(num_nodes: int, rate: float,
                         functional_indices: Optional[np.ndarray] = None,
                         seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Select nodes to corrupt. Returns (corrupt_indices, healthy_indices).

    Uses deterministic seed for reproducibility. Same seed + rate = same nodes.
    """
    rng = np.random.RandomState(seed + int(rate * 100))
    pool = functional_indices if functional_indices is not None else np.arange(num_nodes)
    n_corrupt = max(1, int(len(pool) * rate))
    corrupt = np.sort(rng.choice(pool, n_corrupt, replace=False))
    healthy = np.setdiff1d(pool, corrupt)
    return corrupt, healthy


def load_functional_indices(dataset_name: str, num_nodes: int,
                            node_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Load functional node indices if category files exist. Falls back to all nodes.

    Args:
        dataset_name: Name of the dataset.
        num_nodes: Number of nodes in the (possibly filtered) dataset.
        node_indices: If dataset uses node filtering, the mapping from filtered
            index to original index (e.g., keep_no_dead.npy). Needed to correctly
            identify functional nodes in the filtered space.
    """
    for base in ['datasets/xtraffic', 'datasets']:
        dead_path = os.path.join(base, dataset_name, 'dead_indices.npy')
        major_path = os.path.join(base, dataset_name, 'major_fail_indices.npy')
        if os.path.exists(dead_path):
            dead = np.load(dead_path)
            major = np.load(major_path) if os.path.exists(major_path) else np.array([], dtype=int)
            exclude = set(np.union1d(dead, major).tolist())
            if node_indices is not None:
                # node_indices[i] = original node index for filtered index i
                return np.array([i for i, orig in enumerate(node_indices)
                                 if orig not in exclude])
            return np.setdiff1d(np.arange(num_nodes), exclude)
    return np.arange(num_nodes)
