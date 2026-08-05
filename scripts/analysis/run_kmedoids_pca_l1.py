#!/usr/bin/env python
"""
K-medoids with PCA-50 + L1 distance (MAE-aligned), true GPU FasterPAM.

Pipeline:
  1. Feature extraction → PCA-50
  2. PCA-50 L1 distance matrix on CPU (fast, ~4s)
  3. Move to GPU
  4. True FasterPAM: BUILD + SWAP (O(N^2) per iteration)

Saves indices to k_medoids_pca_l1_euclidean_{ratio}_seed{seed}.json
Overwrites existing files (old broken algorithm results are replaced).
"""
import os, sys, time, json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type
from coreset.fasterpam_gpu import fasterpam_gpu

DATASETS = {
    'SAN_BERNARDINO': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'CONTRA_COSTA': 'baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py',
}
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [42, 123, 456]
PCA_DIM = 50
MAX_ITER = 100


def compute_qc_l1(pca_feat, indices, batch_size=500):
    cs = pca_feat[indices]
    n = len(pca_feat)
    min_d = np.empty(n)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        min_d[i:end] = cdist(pca_feat[i:end], cs, metric='cityblock').min(axis=1)
    return float(np.mean(min_d))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    for ds_name, cfg_path in DATASETS.items():
        print(f"\n{'='*75}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*75}")

        cfg = import_config(cfg_path, verbose=False)
        from basicts.data import TimeSeriesForecastingDataset
        dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
        model_config = cfg['MODEL']
        N = len(dataset)

        print(f"  Extracting features (N={N})...")
        t0 = time.time()
        inputs, targets = extract_features(dataset, model_config)
        raw_feat = get_features_by_type(inputs, targets, 'euclidean')
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca_feat = pca.fit_transform(raw_feat).astype(np.float32)
        print(f"  PCA-{PCA_DIM}: var_explained={pca.explained_variance_ratio_.sum():.3f}, "
              f"time={time.time()-t0:.1f}s")

        print(f"  Computing PCA-50 L1 distance matrix (CPU)...")
        t0 = time.time()
        dist_l1_np = cdist(pca_feat, pca_feat, metric='cityblock').astype(np.float32)
        D = torch.from_numpy(dist_l1_np).to(device)
        print(f"  Distance matrix: {time.time()-t0:.1f}s, moved to {device}")
        del dist_l1_np

        out_dir = f'coreset_indices/{ds_name}'
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  {'ratio':<6s} {'seed':<6s} {'k':<7s} {'cost':>14s} "
              f"{'time(s)':>8s} {'QC_new':>10s} {'QC_stride':>10s} {'QC_old':>10s}")
        print(f"  {'-'*76}")

        for ratio in RATIOS:
            k = int(N * ratio)
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                t0 = time.time()
                medoids = fasterpam_gpu(D, k, init='build', max_iter=MAX_ITER,
                                         seed=seed, verbose=False)
                elapsed = time.time() - t0
                indices = medoids.cpu().tolist()
                cost = D[:, medoids].min(dim=1).values.sum().item()

                ratio_str = f"{ratio:.2f}".replace('.', '')
                out_path = os.path.join(out_dir, f'k_medoids_pca_l1_euclidean_{ratio_str}_seed{seed}.json')
                with open(out_path, 'w') as f:
                    json.dump([int(x) for x in indices], f)

                qc_new = compute_qc_l1(pca_feat, indices)

                stride_path = os.path.join(out_dir, f'stride_euclidean_{ratio_str}_seed{seed}.json')
                qc_stride = compute_qc_l1(pca_feat, json.load(open(stride_path))) \
                            if os.path.exists(stride_path) else float('nan')

                old_path = os.path.join(out_dir, f'k_medoids_euclidean_{ratio_str}_seed{seed}.json')
                qc_old = compute_qc_l1(pca_feat, json.load(open(old_path))) \
                         if os.path.exists(old_path) else float('nan')

                print(f"  {ratio:<6.1f} {seed:<6d} {k:<7d} {cost:>14.1f} "
                      f"{elapsed:>8.1f} {qc_new:>10.3f} {qc_stride:>10.3f} {qc_old:>10.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
