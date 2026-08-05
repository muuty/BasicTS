#!/usr/bin/env python
"""
Compute W₁ distances in raw input space (no PCA).

Outputs:
  - W₁(test, train) via NN proxy
  - W₁(train, coreset; w) ≈ quantization cost for each coreset setting
  - Quick bound check: MAE^w + L_max * (W₁_tt + W₁_tc)

Usage:
    conda activate cuda && python scripts/analysis/compute_w1_raw.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from basicts.data import TimeSeriesForecastingDataset


def extract_flat(ds, channels=None):
    """Extract flattened input features from dataset.

    Args:
        channels: list of channel indices to use (default: [0] = speed only).
            Using fewer channels reduces dimensionality and speeds up cdist.
    """
    if channels is None:
        channels = [0]  # speed only by default
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    all_flat = []
    for batch in loader:
        inp = batch['inputs'].float()
        # Select channels: (B, T, N, C) -> (B, T, N, len(channels))
        inp = inp[..., channels]
        flat = inp.reshape(inp.shape[0], -1).numpy()
        all_flat.append(flat)
    return np.concatenate(all_flat, axis=0)


def nn_w1_proxy(X, Y, batch_size=200):
    """Approximate W₁(X, Y) via average bidirectional NN distance."""
    # X -> Y
    min_x2y = []
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        d = cdist(X[i:end], Y, metric='euclidean')
        min_x2y.append(d.min(axis=1))
    min_x2y = np.concatenate(min_x2y)

    # Y -> X
    min_y2x = []
    for i in range(0, len(Y), batch_size):
        end = min(i + batch_size, len(Y))
        d = cdist(Y[i:end], X, metric='euclidean')
        min_y2x.append(d.min(axis=1))
    min_y2x = np.concatenate(min_y2x)

    return 0.5 * np.mean(min_x2y) + 0.5 * np.mean(min_y2x)


def quantization_cost(train_flat, coreset_indices, batch_size=500):
    """W₁(train, coreset; w) upper bound = (1/n) Σ min_c d(x_i, c)."""
    coreset_feat = train_flat[coreset_indices]
    n = len(train_flat)
    min_dists = np.empty(n)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        d = cdist(train_flat[i:end], coreset_feat, metric='euclidean')
        min_dists[i:end] = d.min(axis=1)
    return float(np.mean(min_dists))


def main():
    outdir = Path('experiments/result/analysis')
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    cfg_path = 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py'
    cfg = import_config(cfg_path, verbose=False)
    ds_params = cfg['DATASET']['PARAM']

    print("Loading datasets...")
    train_ds = TimeSeriesForecastingDataset(mode='train', **ds_params)
    test_ds = TimeSeriesForecastingDataset(mode='test', **ds_params)

    print("Extracting flat features...")
    t0 = time.time()
    train_flat = extract_flat(train_ds)
    test_flat = extract_flat(test_ds)
    print(f"  Train: {train_flat.shape}, Test: {test_flat.shape} ({time.time()-t0:.1f}s)")

    # W₁(test, train) with subsampling
    print("\nComputing W₁(test, train) in raw space...")
    rng = np.random.RandomState(42)
    n_sub = 2000
    train_sub = train_flat[rng.choice(len(train_flat), n_sub, replace=False)]
    test_sub = test_flat[rng.choice(len(test_flat), n_sub, replace=False)]

    t0 = time.time()
    w1_tt_raw = nn_w1_proxy(test_sub, train_sub)
    print(f"  W₁(test, train) raw ≈ {w1_tt_raw:.4f}  ({time.time()-t0:.1f}s)")

    # Per-coreset quantization cost
    print("\nComputing W₁(train, coreset) in raw space...")
    index_base = Path('coreset_indices/SAN_BERNARDINO')
    rows = []

    for idx_file in sorted(index_base.glob('*.json')):
        if idx_file.name == 'proxy_metrics.json':
            continue
        m = re.match(r'^(.+?)_(euclidean|temporal|spatial|combined)_(\d+)_seed(\d+)$',
                     idx_file.stem)
        if not m:
            continue
        method, distance, ratio_pct, seed = m.groups()
        if method not in ('k_medoids', 'k_center', 'graph_cut'):
            continue

        with open(idx_file) as f:
            indices = json.load(f)

        t0 = time.time()
        w1_tc = quantization_cost(train_flat, indices)
        elapsed = time.time() - t0

        rows.append({
            'method': method,
            'distance': distance,
            'ratio': int(ratio_pct) / 100,
            'seed': int(seed),
            'w1_tc_raw': round(w1_tc, 4),
            'w1_tt_raw': round(w1_tt_raw, 4),
            'w1_total_raw': round(w1_tt_raw + w1_tc, 4),
            'k': len(indices),
        })
        print(f"  {method:12s} {distance:10s} r={int(ratio_pct)/100:.2f} "
              f"seed={seed}: W₁_tc={w1_tc:.2f} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    out_path = outdir / 'w1_raw_space.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  W₁(test, train) raw = {w1_tt_raw:.4f}")
    print()
    for method in ['k_medoids', 'k_center', 'graph_cut']:
        s = df[df['method'] == method]
        if len(s) > 0:
            print(f"  {method:12s}: W₁_tc mean={s['w1_tc_raw'].mean():.2f}, "
                  f"total={s['w1_total_raw'].mean():.2f}")

    # Quick bound check with gradient L from earlier
    print(f"\n{'='*60}")
    print("QUICK BOUND CHECK (L from 100-sample gradient test)")
    print(f"{'='*60}")
    L_models = {
        'STGCN': 4.06,
        'AGCRN': 1.34,
        'STAEformer': 0.44,
        'DCRNN': 0.09,
    }
    for mname, L in L_models.items():
        print(f"\n  {mname} (L_max ≈ {L}):")
        for method in ['k_medoids', 'k_center', 'graph_cut']:
            s = df[df['method'] == method]
            if len(s) > 0:
                total_w1 = s['w1_total_raw'].mean()
                bound_term = L * total_w1
                print(f"    {method:12s}: L*W₁ = {L:.2f} × {total_w1:.2f} = {bound_term:.2f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
