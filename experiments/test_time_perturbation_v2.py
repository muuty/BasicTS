"""
Test-time perturbation v2: measure ACTUAL MAE degradation (not just prediction shift).

Key improvements over v1:
  1. Computes MAE(pred_corrupted, target) - MAE(pred_clean, target) = actual degradation
  2. Focuses on realistic noise levels (i=0.05, 0.1, 0.15, 0.2)
  3. Also computes Δ (prediction shift) for comparison
  4. Random node selection (not sorted by flow) to avoid selection bias
  5. Multiple random seeds for node selection to get confidence intervals

Usage:
    python experiments/test_time_perturbation_v2.py --gpu 1
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

# Realistic corruption configs
CONFIGS = [
    # (type, intensity, description)
    ('flow_zero', 1.0, 'Flow=0 (single channel stuck)'),
    ('all_zero', 1.0, 'All channels=0 (complete failure)'),
    ('noisy', 0.05, 'Noise 5% of std (~8 veh, ~6% of mean)'),
    ('noisy', 0.1, 'Noise 10% of std (~16 veh, ~13% of mean)'),
    ('noisy', 0.15, 'Noise 15% of std (~24 veh, ~19% of mean)'),
    ('noisy', 0.2, 'Noise 20% of std (~32 veh, ~25% of mean)'),
    ('noisy', 0.3, 'Noise 30% of std (~47 veh, ~38% of mean)'),
    ('bias', 0.1, 'Bias +16 veh constant offset'),
    ('bias', 0.3, 'Bias +47 veh constant offset'),
    ('spike', 0.5, 'Spikes 50% of std on 50% timesteps'),
]

N_CORRUPT_OPTIONS = [1, 5, 10, 20, 50]
NODE_SELECTION_SEEDS = [42, 123, 456]  # Multiple seeds for confidence


def load_model(device):
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
    rs = get_regular_settings(DATA_NAME)
    return ZScoreScaler(
        dataset_name=DATASET_PATH,
        train_ratio=rs['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=rs['NORM_EACH_CHANNEL'],
        rescale=rs['RESCALE'],
    )


def load_test_dataset():
    rs = get_regular_settings(DATA_NAME)
    return TimeSeriesForecastingDataset(
        dataset_name=DATASET_PATH,
        train_val_test_ratio=rs['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12,
        data_range=(0, 26280),
    )


def apply_corruption(inputs, corrupt_nodes, ctype, intensity, flow_std, rng):
    """Apply corruption to raw inputs (before normalization)."""
    corrupted = inputs.clone()
    B, T, N, C = corrupted.shape

    if ctype == 'flow_zero':
        corrupted[:, :, corrupt_nodes, 0] = 0.0
    elif ctype == 'all_zero':
        for ch in [0, 1, 2]:
            corrupted[:, :, corrupt_nodes, ch] = 0.0
    elif ctype == 'noisy':
        noise = torch.tensor(
            rng.normal(0, intensity * flow_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device
        )
        corrupted[:, :, corrupt_nodes, 0] += noise
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)
    elif ctype == 'bias':
        corrupted[:, :, corrupt_nodes, 0] += intensity * flow_std
    elif ctype == 'spike':
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


def run_inference(model, inputs_norm, device):
    """Run model, return predictions in NORMALIZED space (B, 12, N)."""
    history = inputs_norm[..., FORWARD_FEATURES].to(device)
    future = torch.zeros_like(history)
    future[..., 3] = history[:, -1:, :, 3].expand_as(future[..., 3])
    future[..., 4] = history[:, -1:, :, 4].expand_as(future[..., 4])

    with torch.no_grad():
        pred = model(history_data=history, future_data=future,
                     batch_seen=0, epoch=0, train=False)
    if isinstance(pred, dict):
        pred = pred['prediction']
    if pred.dim() == 4:
        pred = pred[..., 0]
    return pred.cpu()


def run_experiment(model, dataset, scaler, device, functional_nodes):
    """Run all perturbation experiments with actual MAE degradation."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    mean = scaler.mean.float()
    std = scaler.std.float()
    flow_std_val = std.item() if std.dim() == 0 else std.mean().item()

    # Pre-compute clean predictions and targets
    print("Computing clean baseline predictions...")
    all_pred_clean = []
    all_targets = []

    for batch in loader:
        inputs_raw = batch['inputs'].float()
        target_raw = batch['target'].float()

        # Normalize and predict
        inputs_norm = inputs_raw.clone()
        inputs_norm[..., 0] = (inputs_norm[..., 0] - mean) / std
        pred_norm = run_inference(model, inputs_norm, device)

        # Rescale prediction to original
        pred_orig = pred_norm * std + mean  # (B, 12, N)
        target_flow = target_raw[..., 0]    # (B, 12, N) already original scale

        all_pred_clean.append(pred_orig.numpy())
        all_targets.append(target_flow.numpy())

    pred_clean = np.concatenate(all_pred_clean, axis=0)    # (S, 12, N)
    targets = np.concatenate(all_targets, axis=0)           # (S, 12, N)
    print(f"  Clean predictions shape: {pred_clean.shape}")

    # Compute clean per-node masked MAE
    mask = targets > 0
    clean_mae_per_node = np.zeros(NUM_NODES)
    for n in range(NUM_NODES):
        m = mask[:, :, n]
        if m.sum() > 0:
            clean_mae_per_node[n] = np.abs(pred_clean[:, :, n][m] - targets[:, :, n][m]).mean()

    clean_mae_functional = clean_mae_per_node[functional_nodes].mean()
    print(f"  Clean MAE (functional): {clean_mae_functional:.4f}")

    results = {}

    for ctype, intensity, desc in CONFIGS:
        for n_corrupt in N_CORRUPT_OPTIONS:
            seed_results = []

            for seed in NODE_SELECTION_SEEDS:
                rng_select = np.random.RandomState(seed)
                corrupt_nodes = np.sort(rng_select.choice(
                    functional_nodes, min(n_corrupt, len(functional_nodes)), replace=False
                ))
                healthy_functional = np.setdiff1d(functional_nodes, corrupt_nodes)

                rng_corrupt = np.random.RandomState(seed + 1000)

                # Run corrupted inference
                all_pred_corrupted = []
                batch_idx = 0
                for batch in loader:
                    inputs_raw = batch['inputs'].float()

                    # Apply corruption to raw data
                    inputs_corrupted = apply_corruption(
                        inputs_raw, corrupt_nodes, ctype, intensity, flow_std_val, rng_corrupt
                    )

                    # Normalize and predict
                    inputs_corrupted[..., 0] = (inputs_corrupted[..., 0] - mean) / std
                    pred_norm = run_inference(model, inputs_corrupted, device)
                    pred_orig = pred_norm * std + mean
                    all_pred_corrupted.append(pred_orig.numpy())
                    batch_idx += 1

                pred_corrupted = np.concatenate(all_pred_corrupted, axis=0)

                # Compute per-node masked MAE for corrupted
                corrupted_mae_per_node = np.zeros(NUM_NODES)
                for n in range(NUM_NODES):
                    m = mask[:, :, n]
                    if m.sum() > 0:
                        corrupted_mae_per_node[n] = np.abs(
                            pred_corrupted[:, :, n][m] - targets[:, :, n][m]
                        ).mean()

                # Compute prediction shift (Δ)
                delta_per_node = np.abs(pred_corrupted - pred_clean).mean(axis=(0, 1))

                # Metrics on healthy functional nodes
                mae_degradation = corrupted_mae_per_node[healthy_functional] - clean_mae_per_node[healthy_functional]
                delta_healthy = delta_per_node[healthy_functional]

                # Per-node percentile analysis (worst-affected healthy nodes)
                n_healthy = len(healthy_functional)
                pct_counts = {1: max(1, int(n_healthy * 0.01)),
                              5: max(1, int(n_healthy * 0.05)),
                              10: max(1, int(n_healthy * 0.10)),
                              20: max(1, int(n_healthy * 0.20))}

                # Sort descending to get worst-affected
                deg_sorted = np.sort(mae_degradation)[::-1]
                delta_sorted = np.sort(delta_healthy)[::-1]

                pct_metrics = {}
                for pct, cnt in pct_counts.items():
                    pct_metrics[f'degradation_worst_{pct}pct'] = float(deg_sorted[:cnt].mean())
                    pct_metrics[f'delta_worst_{pct}pct'] = float(delta_sorted[:cnt].mean())

                seed_results.append({
                    'healthy_mae_clean': float(clean_mae_per_node[healthy_functional].mean()),
                    'healthy_mae_corrupted': float(corrupted_mae_per_node[healthy_functional].mean()),
                    'healthy_mae_degradation': float(mae_degradation.mean()),
                    'healthy_mae_degradation_max': float(mae_degradation.max()),
                    'healthy_delta': float(delta_per_node[healthy_functional].mean()),
                    'corrupt_mae_corrupted': float(corrupted_mae_per_node[corrupt_nodes].mean()),
                    'corrupt_delta': float(delta_per_node[corrupt_nodes].mean()),
                    'n_healthy_functional': len(healthy_functional),
                    **pct_metrics,
                })

            # Average across seeds
            avg = {}
            for k in seed_results[0]:
                if isinstance(seed_results[0][k], float):
                    vals = [sr[k] for sr in seed_results]
                    avg[k] = float(np.mean(vals))
                    avg[f'{k}_std'] = float(np.std(vals))
                else:
                    avg[k] = seed_results[0][k]

            key = f"{ctype}_i{intensity}_n{n_corrupt}"
            avg['corruption_type'] = ctype
            avg['intensity'] = intensity
            avg['description'] = desc
            avg['n_corrupt'] = n_corrupt
            avg['n_seeds'] = len(NODE_SELECTION_SEEDS)
            results[key] = avg

            deg = avg['healthy_mae_degradation']
            delta = avg['healthy_delta']
            deg_pct = deg / clean_mae_functional * 100
            print(f"  [{key}] MAE degradation: {deg:+.4f} ({deg_pct:+.1f}%)  "
                  f"Δ={delta:.4f}  "
                  f"corrupt_MAE={avg['corrupt_mae_corrupted']:.2f}")

    return results, clean_mae_functional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    print("Loading model...")
    model = load_model(device)

    print("Loading scaler...")
    scaler = load_scaler()

    print("Loading test dataset...")
    dataset = load_test_dataset()

    dead = np.load(f'datasets/{DATASET_PATH}/dead_indices.npy')
    major_fail = np.load(f'datasets/{DATASET_PATH}/major_fail_indices.npy')
    functional = np.setdiff1d(np.arange(NUM_NODES), np.union1d(dead, major_fail))
    print(f"  Functional: {len(functional)}")

    results, clean_mae = run_experiment(model, dataset, scaler, device, functional)

    # Save
    output_dir = 'experiments/perturbation_results'
    os.makedirs(output_dir, exist_ok=True)
    output = {
        'clean_mae_functional': clean_mae,
        'experiments': results,
    }
    with open(os.path.join(output_dir, 'perturbation_v2_mae_degradation.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY: Actual MAE Degradation on Healthy Functional Nodes")
    print(f"  Clean baseline MAE: {clean_mae:.4f}")
    print("=" * 100)
    print(f"{'Config':<30} {'n=1':>12} {'n=5':>12} {'n=10':>12} {'n=20':>12} {'n=50':>12}")
    print("-" * 100)

    seen_configs = []
    for ctype, intensity, desc in CONFIGS:
        label = f"{ctype} i={intensity}"
        if label in seen_configs:
            continue
        seen_configs.append(label)
        row = f"  {label:<28}"
        for n in N_CORRUPT_OPTIONS:
            key = f"{ctype}_i{intensity}_n{n}"
            if key in results:
                deg = results[key]['healthy_mae_degradation']
                pct = deg / clean_mae * 100
                row += f"  {pct:>+8.2f}%  "
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Also print Δ for comparison
    print("\n" + "=" * 100)
    print("COMPARISON: Prediction Shift (Δ) - from v1 metric")
    print("=" * 100)
    print(f"{'Config':<30} {'n=1':>12} {'n=5':>12} {'n=10':>12} {'n=20':>12} {'n=50':>12}")
    print("-" * 100)

    seen_configs = []
    for ctype, intensity, desc in CONFIGS:
        label = f"{ctype} i={intensity}"
        if label in seen_configs:
            continue
        seen_configs.append(label)
        row = f"  {label:<28}"
        for n in N_CORRUPT_OPTIONS:
            key = f"{ctype}_i{intensity}_n{n}"
            if key in results:
                delta = results[key]['healthy_delta']
                pct = delta / clean_mae * 100
                row += f"  {pct:>8.1f}%   "
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Per-node percentile analysis
    for pct_level in [1, 5, 10, 20]:
        for metric, label in [('degradation', 'MAE Degradation'), ('delta', 'Prediction Shift (Δ)')]:
            print("\n" + "=" * 100)
            print(f"WORST {pct_level}% NODES: {label}")
            print("=" * 100)
            print(f"{'Config':<30} {'n=1':>12} {'n=5':>12} {'n=10':>12} {'n=20':>12} {'n=50':>12}")
            print("-" * 100)

            seen_configs = []
            for ctype, intensity, desc in CONFIGS:
                cfg_label = f"{ctype} i={intensity}"
                if cfg_label in seen_configs:
                    continue
                seen_configs.append(cfg_label)
                row = f"  {cfg_label:<28}"
                for n in N_CORRUPT_OPTIONS:
                    key = f"{ctype}_i{intensity}_n{n}"
                    if key in results:
                        val = results[key].get(f'{metric}_worst_{pct_level}pct', 0)
                        if metric == 'degradation':
                            pct = val / clean_mae * 100
                            row += f"  {pct:>+8.2f}%  "
                        else:
                            row += f"  {val:>8.4f}   "
                    else:
                        row += f"{'N/A':>12}"
                print(row)

    print(f"\nResults saved to experiments/perturbation_results/perturbation_v2_mae_degradation.json")


if __name__ == '__main__':
    main()
