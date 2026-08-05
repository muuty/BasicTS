#!/usr/bin/env python
"""
Compute quantization cost (k-medoids objective) for all coreset index files
in multiple feature spaces: euclidean(PCA), temporal(PCA), spatial(PCA), combined(PCA).

Quantization cost = (1/n) Σ min_{s∈S} d(x_i, s)

Saves to proxy_metrics.json with keys:
  quantization_cost          — euclidean PCA-50 (default)
  quantization_median        — euclidean PCA-50
  quantization_max           — euclidean PCA-50
  qc_temporal                — temporal PCA-50
  qc_spatial                 — spatial PCA-50
  qc_combined                — combined PCA-50

Usage:
    sbatch with gpu_rocm partition (needs ~10GB memory for feature extraction)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type

INDEX_BASE = Path('coreset_indices')
DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA': 'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}
PCA_DIM = 50
BATCH_SIZE = 500
DISTANCE_TYPES = ['euclidean', 'temporal', 'spatial', 'combined']


def compute_quantization(features, coreset_indices, batch_size=BATCH_SIZE):
    """Compute (1/n) Σ min_{s∈S} d(x_i, s)."""
    coreset_feat = features[coreset_indices]
    n = features.shape[0]
    min_dists = np.empty(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        dists = cdist(features[i:end], coreset_feat, metric='euclidean')
        min_dists[i:end] = dists.min(axis=1)
        del dists
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

        # Load existing proxy metrics
        with open(pm_path) as f:
            metrics = json.load(f)

        # Find euclidean index entries
        euc_entries = [k for k in metrics if 'euclidean' in k]

        # Check what needs computing
        need_new_distances = [k for k in euc_entries if 'qc_temporal' not in metrics[k]]
        need_euclidean = [k for k in euc_entries if 'quantization_cost' not in metrics[k]]

        print(f"  Euclidean entries: {len(euc_entries)}")
        print(f"  Need euclidean QC: {len(need_euclidean)}")
        print(f"  Need temporal/spatial/combined QC: {len(need_new_distances)}")

        if not need_new_distances and not need_euclidean:
            print("  All done, skipping.")
            continue

        # Extract features once
        print(f"  Loading dataset from {cfg_path}...")
        cfg = import_config(cfg_path, verbose=False)
        from basicts.data import TimeSeriesForecastingDataset
        dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
        model_config = cfg['MODEL']

        t0 = time.time()
        inputs, targets = extract_features(dataset, model_config)
        print(f"  Feature extraction: {time.time()-t0:.1f}s")

        # Build PCA features for each distance type
        pca_cache = {}
        for dt in DISTANCE_TYPES:
            raw = get_features_by_type(inputs, targets, dt)
            n_comp = min(PCA_DIM, raw.shape[1] - 1, raw.shape[0] - 1)
            print(f"  Building {dt} PCA-{n_comp}...", end=" ", flush=True)
            pca = PCA(n_components=n_comp, random_state=42)
            pca_cache[dt] = pca.fit_transform(raw)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"dim={raw.shape[1]} → {n_comp}, var={var_exp:.3f}")
            del raw

        # Compute for each entry
        entries_to_process = set(need_euclidean) | set(need_new_distances)
        total = len(entries_to_process)

        for i, fname in enumerate(sorted(entries_to_process), 1):
            idx_file = idx_dir / fname
            if not idx_file.exists():
                print(f"  [{i}/{total}] SKIP {fname}")
                continue

            with open(idx_file) as f:
                indices = json.load(f)

            t0 = time.time()
            updates = {}

            # Euclidean (if missing)
            if fname in need_euclidean:
                qc = compute_quantization(pca_cache['euclidean'], indices)
                updates['quantization_cost'] = qc['mean']
                updates['quantization_max'] = qc['max']
                updates['quantization_median'] = qc['median']

            # Other distance types (if missing)
            if fname in need_new_distances:
                for dt in ['temporal', 'spatial', 'combined']:
                    qc = compute_quantization(pca_cache[dt], indices)
                    updates[f'qc_{dt}'] = qc['mean']
                    updates[f'qc_{dt}_median'] = qc['median']

            metrics[fname].update(updates)
            elapsed = time.time() - t0

            qc_euc = metrics[fname].get('quantization_cost', 0)
            qc_tmp = metrics[fname].get('qc_temporal', 0)
            qc_spa = metrics[fname].get('qc_spatial', 0)
            print(f"  [{i}/{total}] {fname}: "
                  f"euc={qc_euc:.3f} tmp={qc_tmp:.3f} spa={qc_spa:.3f} "
                  f"({elapsed:.1f}s)")

        # Save
        with open(pm_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved {len(metrics)} entries to {pm_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
