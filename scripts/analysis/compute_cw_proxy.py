"""
Conditional Wasserstein (CW) proxy metric for coreset selections.

Extends qc_raw_l1 (the raw-feature L1 quantization cost) by partitioning
samples into time-of-day / day-of-week bins and computing the quantization
cost WITHIN each bin, then aggregating with bin weights.

Formal proxy
------------
Given conditioning bins $t$ (e.g. time-of-day ∈ {0..23}):

    CW(S) = Σ_t (|T_t|/|T|) * (1/|T_t|) Σ_{x ∈ T_t} min_{s ∈ S_t} ||x - s||_1

    where T_t, S_t are train / coreset samples in bin t. If S_t = ∅,
    fall back to global nearest s ∈ S ("coverage penalty": the missing
    bin contributes the unconditional nearest distance, which is
    typically much larger than the within-bin distance).

Three binnings are computed for comparison:
  - CW_tod    : 24 bins on TOD
  - CW_dow    : 7  bins on DOW
  - CW_joint  : 168 bins on TOD × DOW

Hypothesis
----------
CW should correlate with MAE better than qc_raw_l1 (marginal) because it
penalizes methods that over-cover some bins and under-cover others, even
when their overall marginal distribution looks right. In particular:
  - `recent` should spike on CW_dow (concentrated on last ~9 days)
  - `graph_cut` should spike on CW_tod (anomaly-seeking, TOD-imbalanced)
  - `stride` / `k_medoids` should have low CW on all three binnings

Output
------
- Adds fields `qc_cw_tod`, `qc_cw_dow`, `qc_cw_joint` to each entry in
  `coreset_indices/{DATASET}/proxy_metrics.json`.

Usage
-----
  conda activate cuda && python scripts/analysis/compute_cw_proxy.py
"""

from __future__ import annotations
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
from basicts.data import TimeSeriesForecastingDataset
from coreset.distance import extract_features, get_features_by_type

INDEX_BASE = Path('coreset_indices')
DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA':   'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STEPS_PER_DAY = 288
N_TOD_BINS = 24
N_DOW_BINS = 7


# ---------------------------------------------------------------------------
# Bin assignment
# ---------------------------------------------------------------------------

def bin_assignments(indices: np.ndarray, binning: str) -> np.ndarray:
    """Return bin id for each sample index.

    - 'tod':   24 bins  = (index % 288) // 12
    - 'dow':   7  bins  = (index // 288) % 7
    - 'joint': 168 bins = tod * 7 + dow
    """
    arr = np.asarray(indices)
    tod = (arr % STEPS_PER_DAY) // (STEPS_PER_DAY // N_TOD_BINS)
    dow = (arr // STEPS_PER_DAY) % N_DOW_BINS
    if binning == 'tod':
        return tod
    if binning == 'dow':
        return dow
    if binning == 'joint':
        return tod * N_DOW_BINS + dow
    raise ValueError(f'unknown binning {binning}')


# ---------------------------------------------------------------------------
# Core CW proxy computation
# ---------------------------------------------------------------------------

def compute_cw(
    features: np.ndarray,          # (N, D) raw features for full train set
    coreset_indices: np.ndarray,   # indices into train
    binning: str,
    batch_size: int = 512,
) -> float:
    """Conditional-Wasserstein proxy via within-bin L1 quantization cost.

    For each bin t, computes min_{s ∈ S_t} ||x - s||_1 for every x ∈ T_t.
    Empty bins fall back to global S. Returns the weighted mean.
    """
    n_train = features.shape[0]
    all_train_idx = np.arange(n_train)
    train_bins = bin_assignments(all_train_idx, binning)
    coreset_bins_set = set(bin_assignments(coreset_indices, binning).tolist())
    coreset_feat_full = torch.from_numpy(
        features[coreset_indices]).float().to(DEVICE)

    # Pre-group coreset indices by bin for efficient per-bin lookup
    coreset_idx_by_bin: dict[int, np.ndarray] = {}
    coreset_bins = bin_assignments(coreset_indices, binning)
    for b in np.unique(coreset_bins):
        coreset_idx_by_bin[int(b)] = coreset_indices[coreset_bins == b]

    total_weighted = 0.0
    total_count = 0
    n_missing_bins = 0
    total_train_per_bin = {int(b): int((train_bins == b).sum())
                            for b in np.unique(train_bins)}

    for b, n_t in total_train_per_bin.items():
        if n_t == 0:
            continue
        in_bin = np.where(train_bins == b)[0]
        if b in coreset_idx_by_bin:
            S_b = torch.from_numpy(
                features[coreset_idx_by_bin[b]]).float().to(DEVICE)
        else:
            # Coverage penalty: fall back to global coreset
            S_b = coreset_feat_full
            n_missing_bins += 1

        # For each train sample in bin b, find min L1 distance to S_b
        min_dists = torch.empty(len(in_bin), device=DEVICE)
        for i in range(0, len(in_bin), batch_size):
            end = min(i + batch_size, len(in_bin))
            x = torch.from_numpy(features[in_bin[i:end]]).float().to(DEVICE)
            d = torch.cdist(x, S_b, p=1)   # (batch, |S_b|)
            min_dists[i:end] = d.min(dim=1).values
            del x, d
        bin_mean = min_dists.mean().item()
        total_weighted += bin_mean * n_t
        total_count += n_t

    cw_value = total_weighted / total_count
    return float(cw_value), n_missing_bins


def load_train_features(cfg_path: str) -> np.ndarray:
    """Extract raw train features (flat euclidean representation).
    Mirrors coreset/distance.py → get_features_by_type('euclidean')."""
    cfg = import_config(cfg_path, verbose=False)
    dataset = TimeSeriesForecastingDataset(
        mode='train', **cfg['DATASET']['PARAM'])
    model_config = cfg['MODEL']
    inputs, targets = extract_features(dataset, model_config)
    return get_features_by_type(inputs, targets, 'euclidean').astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    for ds_name, cfg_path in DATASETS.items():
        idx_dir = INDEX_BASE / ds_name
        pm_path = idx_dir / 'proxy_metrics.json'
        print(f'\n{"="*60}\nDataset: {ds_name}\n{"="*60}', flush=True)

        metrics = json.load(open(pm_path))
        # Focus on euclidean entries (matches qc_raw_l1 coverage)
        euc = [k for k in metrics if 'euclidean' in k]
        need = [k for k in euc if not all(
            f'qc_cw_{b}' in metrics[k] for b in ['tod', 'dow', 'joint'])]
        print(f'Euclidean entries: {len(euc)}, need CW: {len(need)}',
              flush=True)
        if not need:
            continue

        print(f'Loading features from {cfg_path} ...', flush=True)
        t0 = time.time()
        features = load_train_features(cfg_path)
        print(f'  feature shape {features.shape}  '
              f'({time.time()-t0:.1f}s)', flush=True)

        for i, fname in enumerate(sorted(need), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                continue
            indices = np.array(json.load(open(idx_file)), dtype=np.int64)

            t0 = time.time()
            cw_tod, miss_tod = compute_cw(features, indices, 'tod')
            cw_dow, miss_dow = compute_cw(features, indices, 'dow')
            cw_joint, miss_j = compute_cw(features, indices, 'joint')
            dt = time.time() - t0

            metrics[fname].update({
                'qc_cw_tod':   round(cw_tod, 3),
                'qc_cw_dow':   round(cw_dow, 3),
                'qc_cw_joint': round(cw_joint, 3),
                'qc_cw_missing_tod':   int(miss_tod),
                'qc_cw_missing_dow':   int(miss_dow),
                'qc_cw_missing_joint': int(miss_j),
            })
            print(f'  [{i}/{len(need)}] {fname}: '
                  f'CW_tod={cw_tod:.1f} CW_dow={cw_dow:.1f} '
                  f'CW_joint={cw_joint:.1f} '
                  f'miss(tod/dow/joint)={miss_tod}/{miss_dow}/{miss_j} '
                  f'({dt:.1f}s)', flush=True)

            # Flush every 10 entries
            if i % 10 == 0:
                with open(pm_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

        with open(pm_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'Saved {len(metrics)} entries to {pm_path}', flush=True)


if __name__ == '__main__':
    main()
