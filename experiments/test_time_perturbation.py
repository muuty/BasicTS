"""
Test-time perturbation experiment: measure contamination propagation.

Given a model trained on clean data, corrupt specific nodes at test time
and measure how much OTHER nodes' predictions change.

This is model-agnostic: works for any STG model (GCN, Transformer, etc.)
regardless of aggregation mechanism (message passing, attention, convolution).

Corruption types:
  - flow_zero:     flow channel = 0 (sensor failure)
  - all_zero:      all physical channels = 0 (complete failure)
  - noisy:         Gaussian noise on flow channel
  - bias:          constant offset on flow channel
  - spike:         random spikes on flow channel

Usage:
    python experiments/test_time_perturbation.py
    python experiments/test_time_perturbation.py --gpu 0
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.STAEformer.arch import STAEformer


# ─── Config ───────────────────────────────────────────────────────────
DATA_NAME = 'SAN_BERNARDINO'
DATASET_PATH = 'xtraffic/SAN_BERNARDINO'
NUM_NODES = 893
CKPT_PATH = 'checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/50637b58eb0e35770d311d9f7bdaa214/STAEformer_best_val_MAE.pt'

FORWARD_FEATURES = [0, 1, 2, 3, 4]
TARGET_FEATURES = [0]

# Corruption types and their parameters
CORRUPTION_CONFIGS = {
    'flow_zero': {
        'description': 'Flow channel stuck at 0 (sensor failure)',
        'intensities': [1.0],  # binary - on/off
    },
    'all_zero': {
        'description': 'All physical channels stuck at 0 (complete failure)',
        'intensities': [1.0],
    },
    'noisy': {
        'description': 'Gaussian noise on flow channel',
        'intensities': [0.1, 0.3, 0.5, 1.0],  # noise_std = intensity * channel_std
    },
    'bias': {
        'description': 'Constant positive bias on flow channel',
        'intensities': [0.1, 0.3, 0.5, 1.0],  # bias = intensity * channel_std
    },
    'spike': {
        'description': 'Random spikes on flow channel (50% of timesteps)',
        'intensities': [0.5, 1.0, 2.0],  # spike magnitude = intensity * channel_std
    },
}

# Number of nodes to corrupt per experiment
N_CORRUPT_OPTIONS = [1, 5, 10, 20, 50]

# Steps per day for time-of-day analysis
STEPS_PER_DAY = 288


def load_model(device):
    """Load the clean baseline model."""
    model = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_scaler():
    """Build scaler for z-score normalization."""
    regular_settings = get_regular_settings(DATA_NAME)
    return ZScoreScaler(
        dataset_name=DATASET_PATH,
        train_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=regular_settings['NORM_EACH_CHANNEL'],
        rescale=regular_settings['RESCALE'],
    )


def load_test_dataset():
    """Load test dataset (clean)."""
    regular_settings = get_regular_settings(DATA_NAME)
    return TimeSeriesForecastingDataset(
        dataset_name=DATASET_PATH,
        train_val_test_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12,
        data_range=(0, 26280),
    )


def load_adj_matrix():
    """Load adjacency matrix for distance computation."""
    adj_path = f'datasets/{DATASET_PATH}/adj_mx.pkl'
    with open(adj_path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    # Handle different pickle formats
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (list, tuple)) and len(obj) == 3:
        return obj[2]
    else:
        return obj


def compute_shortest_paths(adj_mx):
    """Compute shortest path distances using BFS on binary adjacency."""
    n = adj_mx.shape[0]
    # Binary adjacency (connected or not)
    binary_adj = (adj_mx > 0).astype(int)
    np.fill_diagonal(binary_adj, 0)

    dist = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(dist, 0)

    for src in range(n):
        visited = set([src])
        queue = [src]
        d = 0
        while queue:
            next_queue = []
            d += 1
            for node in queue:
                neighbors = np.where(binary_adj[node] > 0)[0]
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        dist[src, nb] = d
                        next_queue.append(nb)
            queue = next_queue

    return dist


def apply_test_corruption(inputs, corrupt_nodes, corruption_type, intensity, flow_std, rng):
    """Apply corruption to specific nodes in test inputs.

    Args:
        inputs: (B, 12, N, 5) raw data tensor (before normalization)
        corrupt_nodes: array of node indices to corrupt
        corruption_type: str
        intensity: float
        flow_std: float, std of flow channel from training data
        rng: numpy random state
    Returns:
        corrupted inputs (cloned)
    """
    corrupted = inputs.clone()
    B, T, N, C = corrupted.shape

    if corruption_type == 'flow_zero':
        corrupted[:, :, corrupt_nodes, 0] = 0.0

    elif corruption_type == 'all_zero':
        corrupted[:, :, corrupt_nodes, 0] = 0.0  # flow
        corrupted[:, :, corrupt_nodes, 1] = 0.0  # occupancy
        corrupted[:, :, corrupt_nodes, 2] = 0.0  # speed

    elif corruption_type == 'noisy':
        noise = torch.tensor(
            rng.normal(0, intensity * flow_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device
        )
        corrupted[:, :, corrupt_nodes, 0] += noise
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)

    elif corruption_type == 'bias':
        corrupted[:, :, corrupt_nodes, 0] += intensity * flow_std
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)

    elif corruption_type == 'spike':
        # Random spikes on 50% of timesteps
        spike_mask = torch.tensor(
            rng.random((B, T, len(corrupt_nodes))) < 0.5,
            dtype=corrupted.dtype, device=corrupted.device
        )
        spike_vals = torch.tensor(
            rng.normal(0, intensity * flow_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device
        )
        corrupted[:, :, corrupt_nodes, 0] += spike_mask * spike_vals
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)

    return corrupted


def run_inference(model, inputs_normalized, device):
    """Run model inference.

    Args:
        inputs_normalized: (B, 12, N, 5) normalized inputs
    Returns:
        predictions: (B, 12, N) in normalized space
    """
    history = inputs_normalized[..., FORWARD_FEATURES].to(device)
    # Create dummy future data (same shape, but flow zeroed out)
    future = torch.zeros_like(history)
    # Copy tod/dow from last timestep (approximation for test-time)
    future[..., 3] = history[:, -1:, :, 3].expand_as(future[..., 3])
    future[..., 4] = history[:, -1:, :, 4].expand_as(future[..., 4])

    with torch.no_grad():
        pred = model(history_data=history, future_data=future,
                     batch_seen=0, epoch=0, train=False)
    if isinstance(pred, dict):
        pred = pred['prediction']
    if pred.dim() == 4:
        pred = pred[..., 0]
    return pred.cpu()  # (B, 12, N)


def get_time_period(sample_idx, total_test_samples, data_range_end=26280):
    """Determine time-of-day period for a test sample.

    Returns: 'morning_peak', 'afternoon', 'evening_peak', 'night'
    """
    regular_settings = get_regular_settings(DATA_NAME)
    ratios = regular_settings['TRAIN_VAL_TEST_RATIO']
    test_start = int(data_range_end * (ratios[0] + ratios[1]))

    # Absolute timestep of this sample
    abs_step = test_start + sample_idx
    tod = abs_step % STEPS_PER_DAY  # 0-287

    hour = tod / 12  # 12 steps per hour
    if 7 <= hour < 10:
        return 'morning_peak'
    elif 10 <= hour < 16:
        return 'midday'
    elif 16 <= hour < 20:
        return 'evening_peak'
    else:
        return 'night'


def run_perturbation_experiment(model, dataset, scaler, device, adj_dist_matrix,
                                 functional_nodes, dead_nodes):
    """Run the full perturbation experiment."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    mean = scaler.mean.float()
    std = scaler.std.float()
    flow_std = std.item() if std.dim() == 0 else std.mean().item()

    rng = np.random.RandomState(42)

    # Select target nodes to corrupt (from functional nodes, spread across traffic levels)
    # Load per-node mean flow for stratification
    desc_path = f'datasets/{DATASET_PATH}/desc.json'
    with open(desc_path, 'r') as f:
        desc = json.load(f)
    data = np.memmap(f'datasets/{DATASET_PATH}/data.dat', dtype='float32',
                     mode='r', shape=tuple(desc['shape']))
    node_mean_flow = data[:, :, 0].mean(axis=0)  # (N,)

    # Sort functional nodes by mean flow, pick evenly spaced ones
    func_sorted = functional_nodes[np.argsort(node_mean_flow[functional_nodes])]
    # Pick 50 evenly spaced functional nodes as corruption targets
    target_pool_size = 50
    indices = np.linspace(0, len(func_sorted) - 1, target_pool_size, dtype=int)
    corruption_target_pool = func_sorted[indices]

    print(f"\nCorruption target pool: {len(corruption_target_pool)} functional nodes")
    print(f"  Flow range: {node_mean_flow[corruption_target_pool].min():.1f} - {node_mean_flow[corruption_target_pool].max():.1f}")

    results = {}

    for ctype, cconfig in CORRUPTION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Corruption type: {ctype} - {cconfig['description']}")

        for intensity in cconfig['intensities']:
            for n_corrupt in N_CORRUPT_OPTIONS:
                if n_corrupt > len(corruption_target_pool):
                    continue

                # Select nodes to corrupt
                corrupt_nodes = corruption_target_pool[:n_corrupt]
                healthy_nodes = np.setdiff1d(np.arange(NUM_NODES), corrupt_nodes)
                healthy_functional = np.intersect1d(healthy_nodes, functional_nodes)

                key = f"{ctype}_i{intensity}_n{n_corrupt}"
                print(f"\n  [{key}] Corrupting {n_corrupt} nodes, intensity={intensity}")

                # Collect per-node prediction deltas
                all_deltas = []  # list of (B, 12, N) arrays
                sample_count = 0
                time_period_deltas = {p: [] for p in ['morning_peak', 'midday', 'evening_peak', 'night']}

                for batch_idx, batch in enumerate(loader):
                    inputs_raw = batch['inputs'].float()    # (B, 12, N, 5)
                    B = inputs_raw.shape[0]

                    # 1. Clean inference
                    inputs_clean = inputs_raw.clone()
                    inputs_clean[..., 0] = (inputs_clean[..., 0] - mean) / std
                    pred_clean = run_inference(model, inputs_clean, device)  # (B, 12, N)

                    # 2. Corrupted inference (corrupt BEFORE normalization)
                    inputs_corrupted = apply_test_corruption(
                        inputs_raw, corrupt_nodes, ctype, intensity, flow_std, rng
                    )
                    inputs_corrupted[..., 0] = (inputs_corrupted[..., 0] - mean) / std
                    pred_corrupted = run_inference(model, inputs_corrupted, device)  # (B, 12, N)

                    # 3. Compute prediction delta (in normalized space)
                    delta = (pred_corrupted - pred_clean).abs()  # (B, 12, N)
                    # Convert to original scale
                    if std.dim() == 0:
                        delta = delta * std.item()
                    else:
                        delta = delta * std.unsqueeze(0).unsqueeze(0)

                    all_deltas.append(delta.numpy())

                    # Time period tracking
                    for b in range(B):
                        period = get_time_period(sample_count + b, len(dataset))
                        time_period_deltas[period].append(delta[b].numpy())

                    sample_count += B

                # Aggregate results
                all_deltas = np.concatenate(all_deltas, axis=0)  # (S, 12, N)
                mean_delta = all_deltas.mean(axis=(0, 1))  # (N,)

                # Distance analysis: for each healthy node, compute min distance to any corrupt node
                if adj_dist_matrix is not None:
                    healthy_dist_to_corrupt = np.array([
                        adj_dist_matrix[h, corrupt_nodes].min()
                        if adj_dist_matrix[h, corrupt_nodes].min() >= 0
                        else -1
                        for h in healthy_functional
                    ])
                else:
                    healthy_dist_to_corrupt = None

                # Per time-period aggregation
                period_results = {}
                for period, deltas in time_period_deltas.items():
                    if deltas:
                        period_arr = np.stack(deltas)  # (S_period, 12, N)
                        period_results[period] = {
                            'mean_delta_healthy_functional': float(period_arr[:, :, healthy_functional].mean()),
                            'max_delta_healthy_functional': float(period_arr[:, :, healthy_functional].mean(axis=(0,1)).max()),
                            'n_samples': len(deltas),
                        }

                # Distance-binned analysis
                dist_analysis = {}
                if healthy_dist_to_corrupt is not None:
                    for max_d in [1, 2, 3, 5, 10]:
                        mask = (healthy_dist_to_corrupt >= 0) & (healthy_dist_to_corrupt <= max_d)
                        if mask.sum() > 0:
                            nodes_in_range = healthy_functional[mask]
                            dist_analysis[f'hop_le_{max_d}'] = {
                                'mean_delta': float(mean_delta[nodes_in_range].mean()),
                                'max_delta': float(mean_delta[nodes_in_range].max()),
                                'n_nodes': int(mask.sum()),
                            }
                    # Also nodes far away (>10 hops or unreachable)
                    far_mask = (healthy_dist_to_corrupt > 10) | (healthy_dist_to_corrupt < 0)
                    if far_mask.sum() > 0:
                        far_nodes = healthy_functional[far_mask]
                        dist_analysis['hop_gt_10'] = {
                            'mean_delta': float(mean_delta[far_nodes].mean()),
                            'max_delta': float(mean_delta[far_nodes].max()),
                            'n_nodes': int(far_mask.sum()),
                        }

                result = {
                    'corruption_type': ctype,
                    'intensity': intensity,
                    'n_corrupt_nodes': n_corrupt,
                    'corrupt_nodes': corrupt_nodes.tolist(),
                    'overall_mean_delta': float(mean_delta.mean()),
                    'healthy_mean_delta': float(mean_delta[healthy_nodes].mean()),
                    'healthy_functional_mean_delta': float(mean_delta[healthy_functional].mean()),
                    'corrupt_mean_delta': float(mean_delta[corrupt_nodes].mean()),
                    'healthy_functional_max_delta': float(mean_delta[healthy_functional].max()),
                    'healthy_functional_median_delta': float(np.median(mean_delta[healthy_functional])),
                    'n_healthy': len(healthy_nodes),
                    'n_healthy_functional': len(healthy_functional),
                    'time_periods': period_results,
                    'distance_analysis': dist_analysis,
                }
                results[key] = result

                print(f"    Healthy functional mean Δ: {result['healthy_functional_mean_delta']:.4f}")
                print(f"    Healthy functional max Δ:  {result['healthy_functional_max_delta']:.4f}")
                print(f"    Corrupt nodes mean Δ:      {result['corrupt_mean_delta']:.4f}")
                if dist_analysis:
                    for dk, dv in dist_analysis.items():
                        print(f"    {dk}: mean_Δ={dv['mean_delta']:.4f} (n={dv['n_nodes']})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    print("Loading model...")
    model = load_model(device)

    print("Loading scaler...")
    scaler = load_scaler()

    print("Loading test dataset...")
    dataset = load_test_dataset()
    print(f"  Test samples: {len(dataset)}")

    print("Loading adjacency matrix and computing shortest paths...")
    adj_mx = load_adj_matrix()
    adj_dist = compute_shortest_paths(adj_mx)
    print(f"  Connected node pairs: {(adj_dist >= 0).sum() - NUM_NODES}")
    print(f"  Max shortest path: {adj_dist[adj_dist >= 0].max()}")

    # Load node categories
    dead = np.load(f'datasets/{DATASET_PATH}/dead_indices.npy')
    major_fail = np.load(f'datasets/{DATASET_PATH}/major_fail_indices.npy')
    problematic = np.union1d(dead, major_fail)
    functional = np.setdiff1d(np.arange(NUM_NODES), problematic)
    print(f"  Functional: {len(functional)}, Dead: {len(dead)}, Major fail: {len(major_fail)}")

    results = run_perturbation_experiment(
        model, dataset, scaler, device, adj_dist, functional, dead
    )

    # Save results
    output_dir = 'experiments/perturbation_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_time_perturbation.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Mean prediction delta on healthy functional nodes")
    print("="*80)
    print(f"{'Experiment':<35} {'Healthy Δ':>10} {'Max Δ':>10} {'Corrupt Δ':>12}")
    print("-"*80)
    for key, r in sorted(results.items()):
        print(f"{key:<35} {r['healthy_functional_mean_delta']:>10.4f} "
              f"{r['healthy_functional_max_delta']:>10.4f} "
              f"{r['corrupt_mean_delta']:>12.4f}")


if __name__ == '__main__':
    main()
