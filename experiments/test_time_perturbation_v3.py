"""
Test-time perturbation v3: per-node impact analysis.

Key improvements over v2:
  1. Per-node Δ and MAE degradation (not just averages)
  2. Top-K most affected nodes analysis
  3. Propagation reach: how many nodes are significantly affected
  4. Focused on key configs for faster execution

Usage:
    python experiments/test_time_perturbation_v3.py --gpu 1
"""

import os
import sys
import json
import argparse
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

# Focused configs (key scenarios only)
CONFIGS = [
    ('flow_zero', 1.0, 'Flow=0 (single channel stuck)'),
    ('all_zero', 1.0, 'All channels=0 (complete failure)'),
    ('noisy', 0.1, 'Noise 10% of std'),
    ('noisy', 0.2, 'Noise 20% of std'),
    ('noisy', 0.3, 'Noise 30% of std'),
    ('spike', 0.5, 'Spikes 50% of std'),
]

N_CORRUPT_OPTIONS = [5, 10, 20, 50]
TOP_K_VALUES = [5, 10, 20, 50]
SEED = 42


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
    corrupted = inputs.clone()
    if ctype == 'flow_zero':
        corrupted[:, :, corrupt_nodes, 0] = 0.0
    elif ctype == 'all_zero':
        for ch in [0, 1, 2]:
            corrupted[:, :, corrupt_nodes, ch] = 0.0
    elif ctype == 'noisy':
        B, T = corrupted.shape[:2]
        noise = torch.tensor(
            rng.normal(0, intensity * flow_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device
        )
        corrupted[:, :, corrupt_nodes, 0] += noise
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)
    elif ctype == 'spike':
        B, T = corrupted.shape[:2]
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
        inputs_norm = inputs_raw.clone()
        inputs_norm[..., 0] = (inputs_norm[..., 0] - mean) / std
        pred_norm = run_inference(model, inputs_norm, device)
        pred_orig = pred_norm * std + mean
        target_flow = target_raw[..., 0]
        all_pred_clean.append(pred_orig.numpy())
        all_targets.append(target_flow.numpy())

    pred_clean = np.concatenate(all_pred_clean, axis=0)  # (S, 12, N)
    targets = np.concatenate(all_targets, axis=0)          # (S, 12, N)
    print(f"  Shape: {pred_clean.shape}")

    # Per-node clean masked MAE
    mask = targets > 0
    clean_mae_per_node = np.zeros(NUM_NODES)
    for n in range(NUM_NODES):
        m = mask[:, :, n]
        if m.sum() > 0:
            clean_mae_per_node[n] = np.abs(pred_clean[:, :, n][m] - targets[:, :, n][m]).mean()

    clean_mae_functional = clean_mae_per_node[functional_nodes].mean()
    print(f"  Clean MAE (functional avg): {clean_mae_functional:.4f}")

    results = {}

    for ctype, intensity, desc in CONFIGS:
        for n_corrupt in N_CORRUPT_OPTIONS:
            rng_select = np.random.RandomState(SEED)
            corrupt_nodes = np.sort(rng_select.choice(
                functional_nodes, min(n_corrupt, len(functional_nodes)), replace=False
            ))
            healthy_functional = np.setdiff1d(functional_nodes, corrupt_nodes)

            rng_corrupt = np.random.RandomState(SEED + 1000)

            # Run corrupted inference
            all_pred_corrupted = []
            for batch in loader:
                inputs_raw = batch['inputs'].float()
                inputs_corrupted = apply_corruption(
                    inputs_raw, corrupt_nodes, ctype, intensity, flow_std_val, rng_corrupt
                )
                inputs_corrupted[..., 0] = (inputs_corrupted[..., 0] - mean) / std
                pred_norm = run_inference(model, inputs_corrupted, device)
                pred_orig = pred_norm * std + mean
                all_pred_corrupted.append(pred_orig.numpy())

            pred_corrupted = np.concatenate(all_pred_corrupted, axis=0)

            # Per-node metrics (on healthy functional only)
            corrupted_mae_per_node = np.zeros(NUM_NODES)
            for n in range(NUM_NODES):
                m = mask[:, :, n]
                if m.sum() > 0:
                    corrupted_mae_per_node[n] = np.abs(
                        pred_corrupted[:, :, n][m] - targets[:, :, n][m]
                    ).mean()

            # Per-node Δ (prediction shift)
            delta_per_node = np.abs(pred_corrupted - pred_clean).mean(axis=(0, 1))  # (N,)

            # Per-node MAE degradation
            degradation_per_node = corrupted_mae_per_node - clean_mae_per_node  # (N,)

            # ── Analysis on healthy functional nodes ──
            h_delta = delta_per_node[healthy_functional]
            h_degrad = degradation_per_node[healthy_functional]
            h_clean_mae = clean_mae_per_node[healthy_functional]

            # Sort by Δ descending → most affected first
            sort_idx = np.argsort(-h_delta)

            entry = {
                'corruption_type': ctype,
                'intensity': intensity,
                'n_corrupt': n_corrupt,
                'n_healthy': len(healthy_functional),
                # System-level (average over all healthy)
                'system_avg_delta': float(h_delta.mean()),
                'system_avg_degradation': float(h_degrad.mean()),
                'system_avg_delta_pct': float(h_delta.mean() / clean_mae_functional * 100),
                'system_avg_degradation_pct': float(h_degrad.mean() / clean_mae_functional * 100),
            }

            # Top-K analysis (ranked by Δ)
            for k in TOP_K_VALUES:
                if k > len(sort_idx):
                    continue
                top_k_idx = sort_idx[:k]
                top_k_delta = h_delta[top_k_idx]
                top_k_degrad = h_degrad[top_k_idx]
                top_k_clean = h_clean_mae[top_k_idx]

                entry[f'top{k}_avg_delta'] = float(top_k_delta.mean())
                entry[f'top{k}_avg_degradation'] = float(top_k_degrad.mean())
                entry[f'top{k}_max_delta'] = float(top_k_delta.max())
                entry[f'top{k}_max_degradation'] = float(top_k_degrad.max())
                # Relative: degradation as % of that node's clean MAE
                rel_degrad = top_k_degrad / np.maximum(top_k_clean, 0.01) * 100
                entry[f'top{k}_avg_rel_degradation_pct'] = float(rel_degrad.mean())
                entry[f'top{k}_max_rel_degradation_pct'] = float(rel_degrad.max())

            # Propagation reach: count nodes above thresholds
            thresholds_abs = [0.1, 0.5, 1.0, 2.0, 5.0]  # absolute MAE units
            for t in thresholds_abs:
                entry[f'reach_delta_gt_{t}'] = int((h_delta > t).sum())
                entry[f'reach_degrad_gt_{t}'] = int((h_degrad > t).sum())

            # Also: fraction of nodes with >1% relative degradation
            rel_degrad_all = h_degrad / np.maximum(h_clean_mae, 0.01) * 100
            for pct_thresh in [1, 5, 10, 20]:
                entry[f'reach_rel_degrad_gt_{pct_thresh}pct'] = int((rel_degrad_all > pct_thresh).sum())

            key = f"{ctype}_i{intensity}_n{n_corrupt}"
            results[key] = entry

            # Print compact summary
            t5_deg = entry.get('top5_avg_degradation', 0)
            t10_deg = entry.get('top10_avg_degradation', 0)
            t5_delta = entry.get('top5_avg_delta', 0)
            reach = entry.get('reach_delta_gt_0.5', 0)
            print(f"  [{key}] sys_deg={entry['system_avg_degradation']:+.3f}  "
                  f"top5_deg={t5_deg:+.3f} top5_Δ={t5_delta:.3f}  "
                  f"top10_deg={t10_deg:+.3f}  "
                  f"reach(Δ>0.5)={reach}")

    return results, clean_mae_functional, clean_mae_per_node


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

    results, clean_mae, clean_mae_per_node = run_experiment(
        model, dataset, scaler, device, functional
    )

    # Save
    output_dir = 'experiments/perturbation_results'
    os.makedirs(output_dir, exist_ok=True)
    output = {
        'clean_mae_functional': clean_mae,
        'experiments': results,
    }
    with open(os.path.join(output_dir, 'perturbation_v3_per_node.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # ── Summary Tables ──
    print("\n" + "=" * 120)
    print("TABLE 1: System-level vs Top-K MAE Degradation")
    print(f"  Clean baseline MAE: {clean_mae:.4f}")
    print("=" * 120)
    print(f"{'Config':<25} {'n_cor':>5} {'SysAvg':>10} {'Top5':>10} {'Top10':>10} {'Top20':>10} {'Top50':>10} {'Reach>0.5':>10}")
    print("-" * 120)

    for ctype, intensity, desc in CONFIGS:
        for n_corrupt in N_CORRUPT_OPTIONS:
            key = f"{ctype}_i{intensity}_n{n_corrupt}"
            if key not in results:
                continue
            r = results[key]
            label = f"{ctype} i={intensity}"
            sys_d = r['system_avg_degradation']
            t5 = r.get('top5_avg_degradation', float('nan'))
            t10 = r.get('top10_avg_degradation', float('nan'))
            t20 = r.get('top20_avg_degradation', float('nan'))
            t50 = r.get('top50_avg_degradation', float('nan'))
            reach = r.get('reach_delta_gt_0.5', 0)
            print(f"  {label:<23} {n_corrupt:>5} {sys_d:>+10.3f} {t5:>+10.3f} {t10:>+10.3f} {t20:>+10.3f} {t50:>+10.3f} {reach:>10}")

    print("\n" + "=" * 120)
    print("TABLE 2: Top-K Prediction Shift (Δ)")
    print("=" * 120)
    print(f"{'Config':<25} {'n_cor':>5} {'SysAvg':>10} {'Top5':>10} {'Top10':>10} {'Top20':>10} {'Top50':>10}")
    print("-" * 120)

    for ctype, intensity, desc in CONFIGS:
        for n_corrupt in N_CORRUPT_OPTIONS:
            key = f"{ctype}_i{intensity}_n{n_corrupt}"
            if key not in results:
                continue
            r = results[key]
            label = f"{ctype} i={intensity}"
            sys_d = r['system_avg_delta']
            t5 = r.get('top5_avg_delta', float('nan'))
            t10 = r.get('top10_avg_delta', float('nan'))
            t20 = r.get('top20_avg_delta', float('nan'))
            t50 = r.get('top50_avg_delta', float('nan'))
            print(f"  {label:<23} {n_corrupt:>5} {sys_d:>10.3f} {t5:>10.3f} {t10:>10.3f} {t20:>10.3f} {t50:>10.3f}")

    print("\n" + "=" * 120)
    print("TABLE 3: Propagation Reach (number of healthy nodes with Δ above threshold)")
    print("=" * 120)
    print(f"{'Config':<25} {'n_cor':>5} {'Δ>0.1':>8} {'Δ>0.5':>8} {'Δ>1.0':>8} {'Δ>2.0':>8} {'Δ>5.0':>8} {'deg>1%':>8} {'deg>5%':>8} {'deg>10%':>8}")
    print("-" * 120)

    for ctype, intensity, desc in CONFIGS:
        for n_corrupt in N_CORRUPT_OPTIONS:
            key = f"{ctype}_i{intensity}_n{n_corrupt}"
            if key not in results:
                continue
            r = results[key]
            label = f"{ctype} i={intensity}"
            print(f"  {label:<23} {n_corrupt:>5} "
                  f"{r.get('reach_delta_gt_0.1', 0):>8} "
                  f"{r.get('reach_delta_gt_0.5', 0):>8} "
                  f"{r.get('reach_delta_gt_1.0', 0):>8} "
                  f"{r.get('reach_delta_gt_2.0', 0):>8} "
                  f"{r.get('reach_delta_gt_5.0', 0):>8} "
                  f"{r.get('reach_rel_degrad_gt_1pct', 0):>8} "
                  f"{r.get('reach_rel_degrad_gt_5pct', 0):>8} "
                  f"{r.get('reach_rel_degrad_gt_10pct', 0):>8}")

    print(f"\nResults saved to {output_dir}/perturbation_v3_per_node.json")


if __name__ == '__main__':
    main()
