"""
Synthetic corruption injection for data quality robustness evaluation.

Three corruption types:
  - channel_stuck: One channel fixed at 0 (e.g., speed=0 while flow/occ normal)
  - noisy: Gaussian noise added to physical channels
  - intermittent_missing: Random timesteps zeroed out (scattered, short)

Usage in dataset config:
    CFG.DATASET.PARAM.corruption = {
        'type': 'channel_stuck',   # 'channel_stuck' | 'noisy' | 'intermittent_missing'
        'rate': 0.1,               # fraction of nodes to corrupt
        'seed': 42,                # for reproducibility
        # Type-specific (optional):
        'stuck_channel': 2,        # channel_stuck: which channel (default: 2=speed)
        'noise_ratio': 0.3,        # noisy: noise_std = ratio * channel_std
        'channels': [0, 1, 2],     # noisy/intermittent: which channels (default: physical)
        'missing_rate': 0.1,       # intermittent_missing: fraction of timesteps per node
    }
"""

import numpy as np
from typing import Dict, List, Optional


def select_corrupt_nodes(num_nodes: int, rate: float, seed: int,
                         exclude_nodes: Optional[List[int]] = None) -> np.ndarray:
    """Select nodes to corrupt. Deterministic for same (seed, rate, exclude_nodes).

    Same rate with different corruption types → same nodes selected.
    """
    rng = np.random.RandomState(seed)
    candidates = np.arange(num_nodes)
    if exclude_nodes is not None:
        candidates = np.setdiff1d(candidates, exclude_nodes)
    n_corrupt = int(len(candidates) * rate)
    return np.sort(rng.choice(candidates, n_corrupt, replace=False))


def apply_corruption(data: np.ndarray, config: Dict) -> np.ndarray:
    """Apply synthetic corruption to traffic data.

    Args:
        data: shape (T, N, C), full time series (will be copied)
        config: corruption configuration dict

    Returns:
        corrupted data (copy), same shape as input
    """
    corruption_type = config['type']
    rate = config['rate']
    seed = config.get('seed', 42)
    exclude_nodes = config.get('exclude_nodes', None)

    num_nodes = data.shape[1]
    corrupt_nodes = select_corrupt_nodes(num_nodes, rate, seed, exclude_nodes)

    if len(corrupt_nodes) == 0:
        return data.copy()

    data = data.copy()

    # Separate RNG for value generation (independent of node selection)
    value_rng = np.random.RandomState(seed + 1000)

    if corruption_type == 'channel_stuck':
        _apply_channel_stuck(data, corrupt_nodes, config)
    elif corruption_type == 'noisy':
        _apply_noisy(data, corrupt_nodes, config, value_rng)
    elif corruption_type == 'intermittent_missing':
        _apply_intermittent_missing(data, corrupt_nodes, config, value_rng)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    return data


def _apply_channel_stuck(data: np.ndarray, nodes: np.ndarray, config: Dict):
    """One channel fixed at 0 for selected nodes. Mimics broken speed sensor."""
    channel = config.get('stuck_channel', 2)  # default: speed
    data[:, nodes, channel] = 0.0


def _apply_noisy(data: np.ndarray, nodes: np.ndarray, config: Dict,
                 rng: np.random.RandomState):
    """Add Gaussian noise to physical channels of selected nodes."""
    noise_ratio = config.get('noise_ratio', 0.3)
    channels = config.get('channels', [0, 1, 2])

    T = data.shape[0]
    for ch in channels:
        ch_std = np.std(data[:, :, ch])
        noise = rng.normal(0, noise_ratio * ch_std, (T, len(nodes)))
        data[:, nodes, ch] += noise.astype(data.dtype)
    # Traffic values are non-negative
    for node in nodes:
        np.clip(data[:, node, :3], 0, None, out=data[:, node, :3])


def _apply_intermittent_missing(data: np.ndarray, nodes: np.ndarray,
                                config: Dict, rng: np.random.RandomState):
    """Randomly zero out scattered timesteps for selected nodes."""
    missing_rate = config.get('missing_rate', 0.1)
    channels = config.get('channels', [0, 1, 2])

    T = data.shape[0]
    n_missing = int(T * missing_rate)
    for node in nodes:
        missing_t = rng.choice(T, n_missing, replace=False)
        for ch in channels:
            data[missing_t, node, ch] = 0.0
