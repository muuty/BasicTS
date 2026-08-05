#!/usr/bin/env python
"""
Run improved k-medoids (FasterPAM) for all ratios and seeds,
then compare QC with stride.
"""
import os, sys, time, json
import numpy as np
import torch
import kmedoids as km
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from coreset.distance import extract_features, compute_distance_matrix, get_features_by_type

DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA': 'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [42, 123, 456]


def qc_pca(pca_feat, indices, batch_size=500):
    cs = pca_feat[indices]
    n = len(pca_feat)
    min_d = np.empty(n)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        min_d[i:end] = cdist(pca_feat[i:end], cs).min(axis=1)
    return np.mean(min_d)


for ds_name, cfg_path in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*70}")

    cfg = import_config(cfg_path, verbose=False)
    from basicts.data import TimeSeriesForecastingDataset
    dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
    model_config = cfg['MODEL']
    N = len(dataset)

    # Compute distance matrix once
    print(f"  Computing distance matrix (N={N})...")
    t0 = time.time()
    inputs, targets = extract_features(dataset, model_config)
    dist_np = compute_distance_matrix(inputs, targets, 'euclidean')
    print(f"  Distance matrix done: {time.time()-t0:.1f}s")

    # PCA features for QC measurement
    raw_feat = get_features_by_type(inputs, targets, 'euclidean')
    pca = PCA(n_components=50, random_state=42)
    pca_feat = pca.fit_transform(raw_feat).astype(np.float32)

    out_dir = f'coreset_indices/{ds_name}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  {'ratio':<6s} {'seed':<6s} {'k':<6s} {'loss':>12s} {'n_swap':>8s} "
          f"{'time(s)':>8s} {'QC_new':>10s} {'QC_stride':>10s} {'QC_old':>10s}")
    print(f"  {'-'*76}")

    for ratio in RATIOS:
        k = int(N * ratio)
        for seed in SEEDS:
            # Run FasterPAM
            t0 = time.time()
            result = km.fasterpam(dist_np, k, max_iter=100, init="build", random_state=seed)
            elapsed = time.time() - t0
            indices = [int(x) for x in result.medoids]

            # Save
            ratio_str = f"{ratio:.2f}".replace('.', '')
            out_path = os.path.join(out_dir, f'k_medoids_v2_euclidean_{ratio_str}_seed{seed}.json')
            with open(out_path, 'w') as f:
                json.dump(indices, f)

            # QC comparison
            qc_new = qc_pca(pca_feat, indices)

            # Load stride
            stride_path = os.path.join(out_dir, f'stride_euclidean_{ratio_str}_seed{seed}.json')
            if os.path.exists(stride_path):
                with open(stride_path) as f:
                    stride_idx = json.load(f)
                qc_stride = qc_pca(pca_feat, stride_idx)
            else:
                qc_stride = float('nan')

            # Load old k-medoids
            old_path = os.path.join(out_dir, f'k_medoids_euclidean_{ratio_str}_seed{seed}.json')
            if os.path.exists(old_path):
                with open(old_path) as f:
                    old_idx = json.load(f)
                qc_old = qc_pca(pca_feat, old_idx)
            else:
                qc_old = float('nan')

            print(f"  {ratio:<6.1f} {seed:<6d} {k:<6d} {result.loss:>12.1f} {result.n_swap:>8d} "
                  f"{elapsed:>8.1f} {qc_new:>10.3f} {qc_stride:>10.3f} {qc_old:>10.3f}")

print("\nDone.")
