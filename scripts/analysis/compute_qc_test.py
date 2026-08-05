#!/usr/bin/env python
"""
Compute quantization cost between coreset (subset of TRAIN) and TEST set.

QC_test = (1/|X_test|) Σ_{x in X_test} min_{s in S_train} ||x - s||_1

This is the theoretically correct proxy for MAE on test set, since the model
is evaluated on test distribution. Standard QC (against train) only measures
how well coreset covers training distribution, not test distribution.

Adds to proxy_metrics.json:
  qc_test_l1, qc_test_l1_median, qc_test_l1_max
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


def extract_features_with_train_stats(dataset, model_config, train_mean, train_std):
    """Extract features from dataset using TRAIN statistics for normalization."""
    n = len(dataset)
    inputs_list = []
    targets_list = []
    for i in range(n):
        sample = dataset[i]
        inp = ((sample['inputs'] - train_mean) / train_std)[:, :, model_config.FORWARD_FEATURES]
        tgt = ((sample['target'] - train_mean) / train_std)[:, :, model_config.TARGET_FEATURES]
        inputs_list.append(inp)
        targets_list.append(tgt)
    return np.array(inputs_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)


def compute_qc_test_l1(test_features, train_features, coreset_indices, batch_size=512):
    """(1/n_test) Σ_{x in test} min_{s in S} ||x - s||_1 via torch.cdist on GPU.

    coreset_indices index INTO train_features.
    """
    coreset_feat = torch.from_numpy(train_features[coreset_indices]).float().to(DEVICE)
    n_test = test_features.shape[0]
    min_dists = torch.empty(n_test, device=DEVICE)
    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        batch = torch.from_numpy(test_features[i:end]).float().to(DEVICE)
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
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for ds_name, cfg_path in DATASETS.items():
        idx_dir = INDEX_BASE / ds_name
        pm_path = idx_dir / 'proxy_metrics.json'

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        with open(pm_path) as f:
            metrics = json.load(f)

        euc_entries = [k for k in metrics if 'euclidean' in k]
        need = [k for k in euc_entries if 'qc_test_l1' not in metrics[k]]
        print(f"  Euclidean entries: {len(euc_entries)}, need: {len(need)}")

        if not need:
            print("  All done, skipping.")
            continue

        # Load TRAIN dataset (for coreset feature lookup)
        print(f"  Loading TRAIN from {cfg_path}...")
        cfg = import_config(cfg_path, verbose=False)
        from basicts.data import TimeSeriesForecastingDataset
        train_dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
        model_config = cfg['MODEL']

        # Compute TRAIN statistics (used for both train and test normalization)
        train_mean = np.mean(train_dataset.data, axis=(0, 1), keepdims=True)
        train_std = np.std(train_dataset.data, axis=(0, 1), keepdims=True)
        train_std[train_std == 0] = 1.0

        t0 = time.time()
        print(f"  Extracting TRAIN features (n={len(train_dataset)})...")
        train_inputs, train_targets = extract_features_with_train_stats(
            train_dataset, model_config, train_mean, train_std)
        train_feat = get_features_by_type(train_inputs, train_targets, 'euclidean').astype(np.float32)
        print(f"  Train features: shape={train_feat.shape}, time={time.time()-t0:.1f}s")

        # Load TEST dataset and apply SAME normalization
        print(f"  Loading TEST...")
        test_dataset = TimeSeriesForecastingDataset(mode='test', **cfg['DATASET']['PARAM'])

        t0 = time.time()
        print(f"  Extracting TEST features (n={len(test_dataset)}) with TRAIN stats...")
        test_inputs, test_targets = extract_features_with_train_stats(
            test_dataset, model_config, train_mean, train_std)
        test_feat = get_features_by_type(test_inputs, test_targets, 'euclidean').astype(np.float32)
        print(f"  Test features: shape={test_feat.shape}, time={time.time()-t0:.1f}s")

        # Free dataset memory
        del train_inputs, train_targets, test_inputs, test_targets

        # Compute QC for each entry
        for i, fname in enumerate(sorted(need), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                print(f"  [{i}/{len(need)}] SKIP {fname}")
                continue

            with open(idx_file) as f:
                indices = json.load(f)

            t0 = time.time()
            qc = compute_qc_test_l1(test_feat, train_feat, indices)
            elapsed = time.time() - t0

            updates = {
                'qc_test_l1': qc['mean'],
                'qc_test_l1_median': qc['median'],
                'qc_test_l1_max': qc['max'],
            }
            metrics[fname].update(updates)
            print(f"  [{i}/{len(need)}] {fname}: "
                  f"L1_test={qc['mean']:.3f}  ({elapsed:.1f}s)")

        # Save after each dataset to avoid losing progress
        with open(pm_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved {len(metrics)} entries to {pm_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
