#!/usr/bin/env python
"""
Why does h_tod add info beyond raw QC?

Test H1: Per-ToD-bin QC — does low h_tod coreset have HIGH variance across bins?
Test H3: Cross-ToD nearest-neighbor — does feature-nearest coreset member share test sample's ToD?

For a selection of coresets (representative), compute:
  - qc_bin[t] = avg over test samples at ToD=t of min distance to coreset
  - tod_match_rate = fraction of test samples whose nearest coreset member has same ToD
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
from coreset.distance import get_features_by_type

INDEX_BASE = Path('coreset_indices')
DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA': 'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ToD binning: 288 timesteps per day, let's use 24 bins (hourly)
N_TOD_BINS = 24
STEPS_PER_DAY = 288
STEPS_PER_BIN = STEPS_PER_DAY // N_TOD_BINS  # 12


def extract_features_and_tod(dataset, model_config, train_mean, train_std):
    """Extract normalized features AND extract ToD of each sample (using first timestep ToD as 'sample ToD')."""
    n = len(dataset)
    feats = []
    tods = []  # ToD of the first (input) timestep — represents when sample starts
    for i in range(n):
        sample = dataset[i]
        inp = ((sample['inputs'] - train_mean) / train_std)[:, :, model_config.FORWARD_FEATURES]
        tgt = ((sample['target'] - train_mean) / train_std)[:, :, model_config.TARGET_FEATURES]
        feat = np.concatenate([inp.reshape(-1), tgt.reshape(-1)])
        feats.append(feat.astype(np.float32))
        # Extract ToD: sample['inputs'] has feature channel 3 = ToD (raw, unnormalized)
        # Use first timestep first node
        tod_raw = sample['inputs'][0, 0, 3]  # ToD of first timestep
        tods.append(tod_raw)
    return np.array(feats), np.array(tods)


def analyze_coreset(test_feat, train_feat, test_tod, train_tod, coreset_indices, batch_size=512):
    """Compute per-ToD-bin QC and ToD-match rate."""
    # Coreset features + their ToDs
    cs_feat = torch.from_numpy(train_feat[coreset_indices]).float().to(DEVICE)
    cs_tod = train_tod[coreset_indices]  # numpy

    # Bin test and coreset by ToD (24 hourly bins)
    # ToD raw is 0..1 (normalized) or 0..287 (stepIdx). Let's check.
    # Looking at data: feature [3] is time of day. Format varies.
    # Let's assume ToD is 0..1 continuous (288 steps → 0..287/288 or 0..1)
    # Or it could be step index 0..287
    # Just bin by value * N_TOD_BINS

    tod_min = min(test_tod.min(), train_tod.min())
    tod_max = max(test_tod.max(), train_tod.max())

    # Normalize to [0, 1) then bin
    tod_range = tod_max - tod_min + 1e-9
    test_bin = np.floor((test_tod - tod_min) / tod_range * N_TOD_BINS).astype(int)
    test_bin = np.clip(test_bin, 0, N_TOD_BINS - 1)
    cs_bin = np.floor((cs_tod - tod_min) / tod_range * N_TOD_BINS).astype(int)
    cs_bin = np.clip(cs_bin, 0, N_TOD_BINS - 1)

    # For each test sample, find nearest coreset member and its distance + ToD
    n_test = test_feat.shape[0]
    nearest_dists = np.empty(n_test, dtype=np.float32)
    nearest_cs_idx = np.empty(n_test, dtype=np.int64)
    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        batch = torch.from_numpy(test_feat[i:end]).float().to(DEVICE)
        dists = torch.cdist(batch, cs_feat, p=1)
        min_v, min_i = dists.min(dim=1)
        nearest_dists[i:end] = min_v.cpu().numpy()
        nearest_cs_idx[i:end] = min_i.cpu().numpy()
        del batch, dists

    # ToD of the nearest coreset member for each test sample
    nearest_tod_bin = cs_bin[nearest_cs_idx]

    # Test H3: ToD matching rate
    tod_match = (nearest_tod_bin == test_bin).mean()

    # Test H1: per-ToD-bin QC
    qc_per_bin = np.empty(N_TOD_BINS)
    n_per_bin = np.empty(N_TOD_BINS, dtype=int)
    for b in range(N_TOD_BINS):
        mask = test_bin == b
        n_per_bin[b] = mask.sum()
        qc_per_bin[b] = nearest_dists[mask].mean() if mask.sum() > 0 else np.nan

    # Coreset's own ToD coverage
    cs_tod_counts = np.bincount(cs_bin, minlength=N_TOD_BINS)
    cs_tod_distribution = cs_tod_counts / cs_tod_counts.sum()
    cs_tod_entropy = -np.sum(cs_tod_distribution * np.log(cs_tod_distribution + 1e-12))
    cs_tod_entropy_normalized = cs_tod_entropy / np.log(N_TOD_BINS)  # normalized to [0,1]

    # Test bin QC statistics
    qc_valid = qc_per_bin[~np.isnan(qc_per_bin)]
    qc_bin_mean = qc_valid.mean()
    qc_bin_std = qc_valid.std()
    qc_bin_max = qc_valid.max()
    qc_bin_min = qc_valid.min()
    qc_bin_range = qc_bin_max - qc_bin_min
    qc_bin_cv = qc_bin_std / qc_bin_mean

    del cs_feat

    return {
        'tod_match_rate': float(tod_match),
        'qc_bin_mean': float(qc_bin_mean),
        'qc_bin_std': float(qc_bin_std),
        'qc_bin_max': float(qc_bin_max),
        'qc_bin_min': float(qc_bin_min),
        'qc_bin_range': float(qc_bin_range),
        'qc_bin_cv': float(qc_bin_cv),
        'qc_per_bin': qc_per_bin.tolist(),
        'n_per_bin': n_per_bin.tolist(),
        'cs_tod_entropy_norm': float(cs_tod_entropy_normalized),
        'cs_tod_distribution': cs_tod_distribution.tolist(),
    }


def main():
    print(f"Device: {DEVICE}")

    for ds_name, cfg_path in DATASETS.items():
        idx_dir = INDEX_BASE / ds_name
        pm_path = idx_dir / 'proxy_metrics.json'

        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        with open(pm_path) as f:
            metrics = json.load(f)

        euc_entries = [k for k in metrics if 'euclidean' in k]
        need = [k for k in euc_entries if 'tod_match_rate' not in metrics[k]]
        print(f"  Euclidean entries: {len(euc_entries)}, need: {len(need)}")
        if not need:
            print("  Done.")
            continue

        # Load TRAIN
        print(f"  Loading TRAIN from {cfg_path}...")
        cfg = import_config(cfg_path, verbose=False)
        from basicts.data import TimeSeriesForecastingDataset
        train_dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
        model_config = cfg['MODEL']

        train_mean = np.mean(train_dataset.data, axis=(0, 1), keepdims=True)
        train_std = np.std(train_dataset.data, axis=(0, 1), keepdims=True)
        train_std[train_std == 0] = 1.0

        t0 = time.time()
        print(f"  Extracting TRAIN features+ToDs (n={len(train_dataset)})...")
        train_feat, train_tod = extract_features_and_tod(
            train_dataset, model_config, train_mean, train_std)
        print(f"    features shape={train_feat.shape}, tod range=[{train_tod.min():.3f}, {train_tod.max():.3f}], time={time.time()-t0:.1f}s")

        test_dataset = TimeSeriesForecastingDataset(mode='test', **cfg['DATASET']['PARAM'])
        t0 = time.time()
        print(f"  Extracting TEST features+ToDs (n={len(test_dataset)})...")
        test_feat, test_tod = extract_features_and_tod(
            test_dataset, model_config, train_mean, train_std)
        print(f"    features shape={test_feat.shape}, tod range=[{test_tod.min():.3f}, {test_tod.max():.3f}], time={time.time()-t0:.1f}s")

        for i, fname in enumerate(sorted(need), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                print(f"  [{i}/{len(need)}] SKIP {fname}")
                continue
            with open(idx_file) as f:
                indices = json.load(f)

            t0 = time.time()
            res = analyze_coreset(test_feat, train_feat, test_tod, train_tod, indices)
            elapsed = time.time() - t0

            # Add to metrics
            for k, v in res.items():
                metrics[fname][k] = v

            print(f"  [{i}/{len(need)}] {fname}: "
                  f"tod_match={res['tod_match_rate']:.3f} "
                  f"qc_bin_cv={res['qc_bin_cv']:.3f} "
                  f"qc_bin_range={res['qc_bin_range']:.1f} "
                  f"cs_tod_H={res['cs_tod_entropy_norm']:.3f} "
                  f"({elapsed:.1f}s)")

        with open(pm_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved {len(metrics)} entries to {pm_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
