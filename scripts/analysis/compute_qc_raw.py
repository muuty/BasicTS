#!/usr/bin/env python
"""
Compute quantization cost with L1 (MAE) distance on RAW features (no PCA).

The downstream loss is MAE (L1), so L1 quantization cost is the theoretically
correct proxy aligned with coreset bounds.

Uses scipy.cdist(metric='cityblock') — optimized C code, no GPU needed.

Adds to proxy_metrics.json:
  qc_raw_l1, qc_raw_l1_median, qc_raw_l1_max
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type

INDEX_BASE = Path('coreset_indices')
DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA': 'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_qc_l1(features, coreset_indices, batch_size=512):
    """(1/n) Σ min_{s∈S} ||x_i - s||_1 via torch.cdist on GPU."""
    coreset_feat = torch.from_numpy(features[coreset_indices]).float().to(DEVICE)
    n = features.shape[0]
    min_dists = torch.empty(n, device=DEVICE)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = torch.from_numpy(features[i:end]).float().to(DEVICE)
        dists = torch.cdist(batch, coreset_feat, p=1)  # L1
        min_dists[i:end] = dists.min(dim=1).values
        del batch, dists
    min_dists = min_dists.cpu().numpy()
    return {
        'mean': float(np.mean(min_dists)),
        'max': float(np.max(min_dists)),
        'median': float(np.median(min_dists)),
    }


def main():
    for ds_name, cfg_path in DATASETS.items():
        idx_dir = INDEX_BASE / ds_name
        pm_path = idx_dir / 'proxy_metrics.json'

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        with open(pm_path) as f:
            metrics = json.load(f)

        euc_entries = [k for k in metrics if 'euclidean' in k]
        need = [k for k in euc_entries if 'qc_raw_l1' not in metrics[k]]
        print(f"  Euclidean entries: {len(euc_entries)}, need: {len(need)}")

        if not need:
            print("  All done, skipping.")
            continue

        print(f"  Loading dataset from {cfg_path}...")
        cfg = import_config(cfg_path, verbose=False)
        from basicts.data import TimeSeriesForecastingDataset
        dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
        model_config = cfg['MODEL']

        t0 = time.time()
        inputs, targets = extract_features(dataset, model_config)
        print(f"  Feature extraction: {time.time()-t0:.1f}s")

        raw_feat = get_features_by_type(inputs, targets, 'euclidean').astype(np.float32)
        print(f"  Raw feature shape: {raw_feat.shape}")

        for i, fname in enumerate(sorted(need), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                print(f"  [{i}/{len(need)}] SKIP {fname}")
                continue

            with open(idx_file) as f:
                indices = json.load(f)

            t0 = time.time()
            qc = compute_qc_l1(raw_feat, indices)
            elapsed = time.time() - t0

            updates = {
                'qc_raw_l1': qc['mean'],
                'qc_raw_l1_median': qc['median'],
                'qc_raw_l1_max': qc['max'],
            }
            metrics[fname].update(updates)
            print(f"  [{i}/{len(need)}] {fname}: "
                  f"L1={qc['mean']:.3f}  ({elapsed:.1f}s)")

        with open(pm_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved {len(metrics)} entries to {pm_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
