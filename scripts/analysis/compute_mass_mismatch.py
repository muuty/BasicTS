#!/usr/bin/env python
"""Representative-mass mismatch of each coreset, in the diagnostic's own geometry.

Proposition 1 bounds the loss discrepancy by the transport cost from the full
training measure to the reduced one.  For a coreset C the smallest achievable
transport cost over all weightings of C is the assignment cost

    J(C) = (1/n) sum_i min_{c in C} ||z_i - c||_1     (already computed: qc_raw_l1)

attained by the assignment-induced weights w_j = |{i : pi(i) = j}| / n.  Training
uses uniform weights u_j = 1/k instead, and the extra transport that costs is
bounded by the coreset diameter times the total variation between w and u:

    W_1(Q_C^w, Q_C^u) <= diam(C) * (1/2) ||w - u||_1.

qc_raw_l1 measures WHERE the retained windows sit.  This script measures the
second term, HOW FAR the mass they represent is from uniform.  Both come out of
the same nearest-assignment pass, so this adds one argmin to the existing
computation.

Motivating case: K-center ranks first on qc_raw_l1 in 6 of 10 dataset-ratio
blocks while ranking fourth on test MAE.  If it also carries a large mass
mismatch, the single-term diagnostic misranks it for a reason the bound names.

Writes into coreset_indices/<DATASET>/proxy_metrics.json:
  mass_tv          (1/2) ||w - u||_1            in [0, 1)
  mass_l1          ||w - u||_1
  mass_chi2        sum_j (w_j - u_j)^2 / u_j    effective-imbalance scale
  mass_gini        Gini coefficient of the counts
  mass_empty_frac  share of retained windows that represent only themselves
  coreset_diam     max pairwise L1 distance within C (subsampled if large)
  mass_bound       coreset_diam * mass_tv       the bound's second term
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
DIAM_SAMPLE = 2048


def assignment_counts(features, coreset_indices, batch_size=256):
    """One nearest-assignment pass: mean min distance and the assignment histogram."""
    core = torch.from_numpy(features[coreset_indices]).float().to(DEVICE)
    n, k = features.shape[0], len(coreset_indices)
    counts = torch.zeros(k, dtype=torch.float64, device=DEVICE)
    total = 0.0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = torch.from_numpy(features[i:end]).float().to(DEVICE)
        d = torch.cdist(batch, core, p=1)
        mn, am = d.min(dim=1)
        total += mn.double().sum().item()
        counts += torch.bincount(am, minlength=k).double()
    w = (counts / counts.sum()).cpu().numpy()
    return total / n, w


def coreset_diameter(features, coreset_indices, rng):
    idx = np.asarray(coreset_indices)
    if len(idx) > DIAM_SAMPLE:
        idx = rng.choice(idx, size=DIAM_SAMPLE, replace=False)
    x = torch.from_numpy(features[idx]).float().to(DEVICE)
    best = 0.0
    for i in range(0, len(idx), 256):
        d = torch.cdist(x[i:i + 256], x, p=1)
        best = max(best, float(d.max().item()))
    return best


def gini(w):
    v = np.sort(w)
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def main():
    rng = np.random.default_rng(42)
    for ds_name, cfg_path in DATASETS.items():
        idx_dir = INDEX_BASE / ds_name
        pm_path = idx_dir / 'proxy_metrics.json'
        with open(pm_path) as f:
            metrics = json.load(f)

        euc = [k for k in metrics if 'euclidean' in k]
        need = [k for k in euc if 'mass_tv' not in metrics[k]]
        print(f"\n=== {ds_name}: {len(euc)} euclidean entries, {len(need)} to do", flush=True)
        if not need:
            continue

        cfg = import_config(cfg_path, verbose=False)
        from basicts.data import TimeSeriesForecastingDataset
        dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
        inputs, targets = extract_features(dataset, cfg['MODEL'])
        feat = get_features_by_type(inputs, targets, 'euclidean').astype(np.float32)
        print(f"  features {feat.shape} on {DEVICE}", flush=True)

        for i, fname in enumerate(sorted(need), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                print(f"  [{i}/{len(need)}] SKIP {fname} (no index file)", flush=True)
                continue
            with open(idx_file) as f:
                indices = json.load(f)

            t0 = time.time()
            qc, w = assignment_counts(feat, indices)
            k = len(w)
            u = 1.0 / k
            l1 = float(np.abs(w - u).sum())
            diam = coreset_diameter(feat, indices, rng)
            metrics[fname].update({
                'mass_tv': round(l1 / 2, 6),
                'mass_l1': round(l1, 6),
                'mass_chi2': round(float(((w - u) ** 2).sum() / u), 6),
                'mass_gini': round(gini(w), 6),
                'mass_empty_frac': round(float((w <= 1.0 / len(feat) + 1e-12).mean()), 6),
                'coreset_diam': round(diam, 4),
                'mass_bound': round(diam * l1 / 2, 4),
                'qc_recheck': round(qc, 6),
            })
            print(f"  [{i}/{len(need)}] {fname}: TV={l1/2:.4f} gini={gini(w):.3f} "
                  f"diam={diam:.1f} qc={qc:.3f} ({time.time()-t0:.1f}s)", flush=True)

            with open(pm_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        print(f"  saved {pm_path}", flush=True)
    print("\nDone.")


if __name__ == '__main__':
    main()
