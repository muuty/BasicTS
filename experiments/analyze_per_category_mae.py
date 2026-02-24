"""
Per-node MAE analysis by sensor quality (flow_zero_rate category).

Computes masked MAE (zero targets excluded) and unmasked MAE per node,
grouped by flow_zero_rate categories:
  - Dead (>90% zero): sensor failure
  - Major fail (50-90%): severe data loss
  - Partial fail (5-50%): intermittent data loss
  - Functional (<5%): normal sensors

Usage:
    python experiments/analyze_per_category_mae.py
    python experiments/analyze_per_category_mae.py --save  # save results to JSON
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def compute_flow_zero_rate(data_path='datasets/xtraffic/SAN_BERNARDINO', data_range=(0, 26280)):
    """Compute per-node flow zero rate."""
    with open(f'{data_path}/desc.json') as f:
        desc = json.load(f)
    data = np.memmap(f'{data_path}/data.dat', dtype='float32', mode='r', shape=tuple(desc['shape']))
    flow = data[data_range[0]:data_range[1], :, 0]
    return (flow == 0).mean(axis=0)


def categorize_nodes(flow_zero_rate):
    """Categorize nodes by flow zero rate."""
    return {
        'Dead (>90%)': flow_zero_rate > 0.9,
        'Major fail (50-90%)': (flow_zero_rate > 0.5) & (flow_zero_rate <= 0.9),
        'Partial fail (5-50%)': (flow_zero_rate > 0.05) & (flow_zero_rate <= 0.5),
        'Functional (<5%)': flow_zero_rate <= 0.05,
    }


def compute_per_node_mae(pred_path, tgt_path, n_samples, output_len, n_nodes):
    """Compute per-node masked and unmasked MAE from memmap files."""
    shape = (n_samples, output_len, n_nodes)
    pred = np.memmap(pred_path, dtype='float32', mode='r', shape=shape)
    tgt = np.memmap(tgt_path, dtype='float32', mode='r', shape=shape)

    raw_sum = np.zeros(n_nodes, dtype='float64')
    masked_sum = np.zeros(n_nodes, dtype='float64')
    masked_count = np.zeros(n_nodes, dtype='float64')

    chunk = 500
    for i in range(0, n_samples, chunk):
        end = min(i + chunk, n_samples)
        p = np.array(pred[i:end])
        t = np.array(tgt[i:end])
        ae = np.abs(p - t)

        raw_sum += ae.sum(axis=(0, 1))

        mask = t != 0
        masked_sum += (ae * mask).sum(axis=(0, 1))
        masked_count += mask.sum(axis=(0, 1))

    raw_mae = raw_sum / (n_samples * output_len)
    safe_count = np.maximum(masked_count, 1)
    masked_mae = masked_sum / safe_count

    return raw_mae, masked_mae


def find_experiments(checkpoints_dir='checkpoints'):
    """Find all experiments with saved test_results."""
    experiments = {}
    for pred_path in Path(checkpoints_dir).rglob('test_results/predictions.npy'):
        test_results_dir = pred_path.parent
        ckpt_dir = test_results_dir.parent

        # Get experiment name from top-level checkpoint dir
        parts = pred_path.relative_to(checkpoints_dir).parts
        exp_name = parts[0]

        # Also check test_metrics.json exists
        metrics_path = ckpt_dir / 'test_metrics.json'
        if not metrics_path.exists():
            continue

        experiments[exp_name] = {
            'pred_path': str(pred_path),
            'tgt_path': str(test_results_dir / 'targets.npy'),
            'metrics_path': str(metrics_path),
            'ckpt_dir': str(ckpt_dir),
        }

    return experiments


def main():
    parser = argparse.ArgumentParser(description='Per-category MAE analysis')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    parser.add_argument('--checkpoints-dir', default='checkpoints', help='Checkpoints directory')
    args = parser.parse_args()

    # 1. Compute flow zero rate and categories
    flow_zero_rate = compute_flow_zero_rate()
    categories = categorize_nodes(flow_zero_rate)
    n_nodes = len(flow_zero_rate)

    print("Sensor categories:")
    for name, mask in categories.items():
        print(f"  {name}: {mask.sum()} nodes")

    # 2. Find experiments
    experiments = find_experiments(args.checkpoints_dir)
    print(f"\nFound {len(experiments)} experiments with saved predictions\n")

    # 3. Compute per-node MAE for each
    # Shape params (all same for SAN_BERNARDINO 3mo)
    N_SAMPLES = 5233
    OUTPUT_LEN = 12

    results = {}
    for exp_name in sorted(experiments.keys()):
        info = experiments[exp_name]

        # Load test_metrics for overall MAE
        with open(info['metrics_path']) as f:
            metrics = json.load(f)

        raw_mae, masked_mae = compute_per_node_mae(
            info['pred_path'], info['tgt_path'], N_SAMPLES, OUTPUT_LEN, n_nodes
        )

        # Per-category stats
        cat_stats = {}
        for cat_name, mask in categories.items():
            cat_stats[cat_name] = {
                'n': int(mask.sum()),
                'raw_mae': float(raw_mae[mask].mean()),
                'masked_mae': float(masked_mae[mask].mean()),
            }

        results[exp_name] = {
            'overall_mae': metrics['overall']['MAE'],
            'overall_raw_mae': float(raw_mae.mean()),
            'overall_masked_mae': float(masked_mae.mean()),
            'pn_w1': metrics['robustness']['per_node']['worst_1pct_MAE'],
            'categories': cat_stats,
        }
        print(f"  {exp_name}: MAE={metrics['overall']['MAE']:.2f}")

    # 4. Print comprehensive table - MASKED MAE
    print("\n" + "=" * 130)
    print("MASKED MAE BY CATEGORY (zero targets excluded)")
    print("=" * 130)

    cat_names = list(categories.keys())
    header = f"{'Experiment':<55} | {'MAE':>6} | {'pn_w1':>6}"
    for cn in cat_names:
        short = cn.split('(')[0].strip()
        header += f" | {short:>12}"
    header += f" | {'ALL(masked)':>12}"
    print(header)
    print("-" * 130)

    for exp_name in sorted(results.keys()):
        r = results[exp_name]
        row = f"{exp_name:<55} | {r['overall_mae']:>6.2f} | {r['pn_w1']:>6.2f}"
        for cn in cat_names:
            row += f" | {r['categories'][cn]['masked_mae']:>12.2f}"
        row += f" | {r['overall_masked_mae']:>12.2f}"
        print(row)

    # 5. Print RAW MAE table
    print("\n" + "=" * 130)
    print("RAW (UNMASKED) MAE BY CATEGORY")
    print("=" * 130)

    header = f"{'Experiment':<55} | {'MAE':>6} | {'pn_w1':>6}"
    for cn in cat_names:
        short = cn.split('(')[0].strip()
        header += f" | {short:>12}"
    header += f" | {'ALL(raw)':>12}"
    print(header)
    print("-" * 130)

    for exp_name in sorted(results.keys()):
        r = results[exp_name]
        row = f"{exp_name:<55} | {r['overall_mae']:>6.2f} | {r['pn_w1']:>6.2f}"
        for cn in cat_names:
            row += f" | {r['categories'][cn]['raw_mae']:>12.2f}"
        row += f" | {r['overall_raw_mae']:>12.2f}"
        print(row)

    # 6. Save results
    if args.save:
        out_path = 'experiments/per_category_mae_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
