"""
Analyze corruption experiment results: healthy vs corrupted node MAE.

Loads best checkpoint for each completed experiment, runs test inference,
and computes per-node MAE split by healthy/corrupted nodes.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.data.corruption import select_corrupt_nodes
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.STAEformer.arch import STAEformer


def load_model(ckpt_dir, device):
    """Load best model from checkpoint directory."""
    model = STAEformer(
        num_nodes=893, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    # Find best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'STAEformer_best_val_MAE.pt')
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def run_test(model, dataset, scaler, device, batch_size=64):
    """Run test inference and return per-node MAE."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []

    mean = scaler.mean.float()
    std = scaler.std.float()

    with torch.no_grad():
        for batch in loader:
            inputs = batch['inputs'].float().to(device)      # (B, 12, N, 5)
            target = batch['target'].float().to(device)       # (B, 12, N, 5)

            # Z-normalize flow channel (ch 0) like the runner does
            m, s = mean.to(device), std.to(device)
            inputs[..., 0] = (inputs[..., 0] - m) / s
            target[..., 0] = (target[..., 0] - m) / s

            # Select forward features and run model
            history = inputs[..., [0, 1, 2, 3, 4]]
            future = target[..., [0, 1, 2, 3, 4]]
            # Clear future flow for test (like runner does)
            future[..., 0] = torch.empty_like(future[..., 0])

            pred = model(history, future, batch_seen=0, epoch=0, train=False)
            if isinstance(pred, dict):
                pred = pred['prediction']

            # pred: (B, 12, N, 1) or (B, 12, N) - in normalized space
            if pred.dim() == 4:
                pred = pred[..., 0]
            target_flow_norm = target[..., 0]  # normalized

            # Inverse z-score
            pred_rescaled = (pred * s + m).cpu()
            target_rescaled = (target_flow_norm * s + m).cpu()

            all_preds.append(pred_rescaled.numpy())
            all_targets.append(target_rescaled.numpy())

    preds = np.concatenate(all_preds, axis=0)    # (S, 12, N)
    targets = np.concatenate(all_targets, axis=0)  # (S, 12, N)

    # Per-node masked MAE (exclude zero targets)
    errors = np.abs(preds - targets)  # (S, 12, N)
    mask = targets > 0  # non-zero targets

    per_node_mae = np.zeros(893)
    for n in range(893):
        m = mask[:, :, n]
        if m.sum() > 0:
            per_node_mae[n] = errors[:, :, n][m].mean()
        else:
            per_node_mae[n] = 0.0  # dead node

    return per_node_mae, preds, targets


def analyze_experiment(ckpt_dir, corruption_config, device):
    """Analyze one experiment."""
    # Build scaler
    DATA_NAME = 'xtraffic/SAN_BERNARDINO'
    regular_settings = get_regular_settings('SAN_BERNARDINO')
    scaler = ZScoreScaler(
        dataset_name=DATA_NAME,
        train_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=regular_settings['NORM_EACH_CHANNEL'],
        rescale=regular_settings['RESCALE'],
    )

    # Build test dataset with corruption
    dataset = TimeSeriesForecastingDataset(
        dataset_name=DATA_NAME,
        train_val_test_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12,
        data_range=(0, 26280),
        corruption=corruption_config,
    )

    # Get corrupted nodes
    if corruption_config is not None:
        corrupt_nodes = select_corrupt_nodes(
            893, corruption_config['rate'],
            corruption_config.get('seed', 42),
        )
    else:
        corrupt_nodes = np.array([], dtype=int)

    healthy_nodes = np.setdiff1d(np.arange(893), corrupt_nodes)

    # Load model and run test
    # Find the hash subdirectory
    subdirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    full_ckpt_dir = os.path.join(ckpt_dir, subdirs[0])

    model = load_model(full_ckpt_dir, device)
    per_node_mae, _, _ = run_test(model, dataset, scaler, device)

    # Also load dead/functional node categories
    dead_indices = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
    major_fail = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')
    problematic = np.union1d(dead_indices, major_fail)
    functional = np.setdiff1d(np.arange(893), problematic)

    # Splits
    healthy_functional = np.intersect1d(healthy_nodes, functional)
    corrupt_functional = np.intersect1d(corrupt_nodes, functional)

    result = {
        'overall_mae': float(per_node_mae.mean()),
        'healthy_all_mae': float(per_node_mae[healthy_nodes].mean()) if len(healthy_nodes) > 0 else None,
        'corrupt_all_mae': float(per_node_mae[corrupt_nodes].mean()) if len(corrupt_nodes) > 0 else None,
        'healthy_functional_mae': float(per_node_mae[healthy_functional].mean()) if len(healthy_functional) > 0 else None,
        'corrupt_functional_mae': float(per_node_mae[corrupt_functional].mean()) if len(corrupt_functional) > 0 else None,
        'n_healthy': len(healthy_nodes),
        'n_corrupt': len(corrupt_nodes),
        'n_healthy_functional': len(healthy_functional),
        'n_corrupt_functional': len(corrupt_functional),
    }
    return result


def main():
    device = torch.device('cuda:0')

    experiments = [
        ('baseline (no corruption)', 'checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12', None),
        ('channel_stuck r=10%', 'checkpoints/STAEformer_5ch_corrupt_channel_stuck_r10/SAN_BERNARDINO_30_12_12',
         {'type': 'channel_stuck', 'rate': 0.1, 'seed': 42, 'stuck_channel': 2}),
        ('channel_stuck r=20%', 'checkpoints/STAEformer_5ch_corrupt_channel_stuck_r20/SAN_BERNARDINO_30_12_12',
         {'type': 'channel_stuck', 'rate': 0.2, 'seed': 42, 'stuck_channel': 2}),
        ('channel_stuck r=30%', 'checkpoints/STAEformer_5ch_corrupt_channel_stuck_r30/SAN_BERNARDINO_30_12_12',
         {'type': 'channel_stuck', 'rate': 0.3, 'seed': 42, 'stuck_channel': 2}),
    ]

    # Only analyze experiments that exist
    experiments = [(name, path, cfg) for name, path, cfg in experiments
                   if os.path.exists(path)]

    results = {}
    for name, ckpt_dir, corruption_config in experiments:
        print(f"\n=== {name} ===")
        result = analyze_experiment(ckpt_dir, corruption_config, device)
        results[name] = result

        print(f"  Overall MAE:              {result['overall_mae']:.4f}")
        print(f"  Healthy nodes MAE:        {result['healthy_all_mae']:.4f} (n={result['n_healthy']})")
        if result['corrupt_all_mae'] is not None:
            print(f"  Corrupt nodes MAE:        {result['corrupt_all_mae']:.4f} (n={result['n_corrupt']})")
        print(f"  Healthy functional MAE:   {result['healthy_functional_mae']:.4f} (n={result['n_healthy_functional']})")
        if result['corrupt_functional_mae'] is not None:
            print(f"  Corrupt functional MAE:   {result['corrupt_functional_mae']:.4f} (n={result['n_corrupt_functional']})")

    # Summary table
    print("\n\n=== SUMMARY: Healthy Functional Node MAE ===")
    print(f"{'Experiment':<30} {'Overall':>10} {'Healthy':>10} {'Corrupt':>10} {'Δ Healthy':>10}")
    baseline_healthy = results.get('baseline (no corruption)', {}).get('healthy_functional_mae', 0)
    for name, result in results.items():
        healthy = result['healthy_functional_mae']
        corrupt = result['corrupt_functional_mae'] if result['corrupt_functional_mae'] else '-'
        delta = f"{healthy - baseline_healthy:+.4f}" if baseline_healthy else '-'
        corrupt_str = f"{corrupt:.4f}" if isinstance(corrupt, float) else corrupt
        print(f"{name:<30} {result['overall_mae']:>10.4f} {healthy:>10.4f} {corrupt_str:>10} {delta:>10}")

    # Save results
    output_path = 'experiments/corruption_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
