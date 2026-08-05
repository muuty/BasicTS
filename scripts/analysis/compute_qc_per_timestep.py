#!/usr/bin/env python
"""
Compute QC per-timestep to test the "flat QC loses multi-dim structure" hypothesis.

For each sample, feature is (24, nodes, F) = concat(12 input, 12 output).
Instead of flattening all at once, compute QC at each timestep slice separately.

QC^(t)(S) = (1/|X_test|) Σ_{x_test} min_{s in S} ||x_test^(t) - s^(t)||_1

where x^(t) is the (nodes, F)-dim slice at timestep t.

Outputs:
  - For each coreset: qc_test_per_timestep = [qc_t for t in 0..23]
  - Can compare variance across t, correlation with MAE, etc.
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

INDEX_BASE = Path('coreset_indices')
DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA': 'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_features_structured(dataset, model_config, train_mean, train_std):
    """Extract features WITHOUT flattening — keep (T, nodes, F) structure."""
    n = len(dataset)
    all_inputs = []
    all_targets = []
    for i in range(n):
        sample = dataset[i]
        inp = ((sample['inputs'] - train_mean) / train_std)[:, :, model_config.FORWARD_FEATURES]
        tgt = ((sample['target'] - train_mean) / train_std)[:, :, model_config.TARGET_FEATURES]
        all_inputs.append(inp)
        all_targets.append(tgt)
    # Stack: (N, T_in, nodes, F_in), (N, T_out, nodes, F_out)
    inp_arr = np.stack(all_inputs, axis=0).astype(np.float32)
    tgt_arr = np.stack(all_targets, axis=0).astype(np.float32)
    return inp_arr, tgt_arr


def compute_qc_per_timestep(test_timesteps, train_timesteps, coreset_indices, batch_size=512):
    """For each timestep t, compute QC_t.

    test_timesteps: (N_test, T, nodes*F) — already flattened over nodes*F per timestep
    train_timesteps: (N_train, T, nodes*F)
    Returns list of T QC values.
    """
    T = test_timesteps.shape[1]
    qc_per_t = []
    # Coreset features per timestep
    coreset_per_t = train_timesteps[coreset_indices]  # (|S|, T, nodes*F)

    for t in range(T):
        cs_feat = torch.from_numpy(coreset_per_t[:, t, :]).float().to(DEVICE)
        n_test = test_timesteps.shape[0]
        min_dists = torch.empty(n_test, device=DEVICE)
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            batch = torch.from_numpy(test_timesteps[i:end, t, :]).float().to(DEVICE)
            dists = torch.cdist(batch, cs_feat, p=1)
            min_dists[i:end] = dists.min(dim=1).values
            del batch, dists
        qc_per_t.append(float(min_dists.mean().cpu()))
        del cs_feat
    return qc_per_t


def main():
    print(f"Device: {DEVICE}")

    for ds_name, cfg_path in DATASETS.items():
        idx_dir = INDEX_BASE / ds_name
        pm_path = idx_dir / 'proxy_metrics.json'

        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        with open(pm_path) as f:
            metrics = json.load(f)

        euc_entries = [k for k in metrics if 'euclidean' in k]
        need = [k for k in euc_entries if 'qc_test_per_t' not in metrics[k]]
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
        print(f"  Extracting TRAIN features (n={len(train_dataset)})...")
        train_inp, train_tgt = extract_features_structured(
            train_dataset, model_config, train_mean, train_std)
        print(f"    inputs={train_inp.shape}, targets={train_tgt.shape}, time={time.time()-t0:.1f}s")

        # Merge input + target across time axis
        # inputs: (N, T_in, nodes, F_in); targets: (N, T_out, nodes, F_out)
        # Full temporal axis: T_in + T_out = 24
        # But F_in != F_out generally. Handle by flattening (nodes, F) per timestep.
        N, T_in, nodes, F_in = train_inp.shape
        _, T_out, _, F_out = train_tgt.shape
        train_inp_flat = train_inp.reshape(N, T_in, nodes * F_in)  # (N, T_in, D_in)
        train_tgt_flat = train_tgt.reshape(N, T_out, nodes * F_out)  # (N, T_out, D_out)

        # Load TEST
        print(f"  Loading TEST...")
        test_dataset = TimeSeriesForecastingDataset(mode='test', **cfg['DATASET']['PARAM'])

        t0 = time.time()
        print(f"  Extracting TEST features (n={len(test_dataset)})...")
        test_inp, test_tgt = extract_features_structured(
            test_dataset, model_config, train_mean, train_std)
        N_test, _, _, _ = test_inp.shape
        test_inp_flat = test_inp.reshape(N_test, T_in, nodes * F_in)
        test_tgt_flat = test_tgt.reshape(N_test, T_out, nodes * F_out)
        print(f"    time={time.time()-t0:.1f}s")

        del train_inp, train_tgt, test_inp, test_tgt

        # Compute per-timestep QC for each needed entry
        for i, fname in enumerate(sorted(need), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                print(f"  [{i}/{len(need)}] SKIP {fname}")
                continue
            with open(idx_file) as f:
                indices = json.load(f)

            t0 = time.time()
            # Input timesteps QC
            qc_input = compute_qc_per_timestep(test_inp_flat, train_inp_flat, indices)
            # Target timesteps QC
            qc_target = compute_qc_per_timestep(test_tgt_flat, train_tgt_flat, indices)
            elapsed = time.time() - t0

            metrics[fname]['qc_test_per_t_input'] = qc_input
            metrics[fname]['qc_test_per_t_target'] = qc_target
            metrics[fname]['qc_test_per_t'] = qc_input + qc_target  # concat for convenience

            mean_qc = np.mean(qc_input + qc_target)
            std_qc = np.std(qc_input + qc_target)
            print(f"  [{i}/{len(need)}] {fname}: mean_qc_t={mean_qc:.2f} std_qc_t={std_qc:.2f} range=[{min(qc_input+qc_target):.1f}, {max(qc_input+qc_target):.1f}]  ({elapsed:.1f}s)")

        with open(pm_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved {len(metrics)} entries to {pm_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
