#!/usr/bin/env python
"""
Quick test: run improved k-medoids (FasterPAM via Rust) and compare QC with stride.
"""
import os, sys, time, json
import numpy as np
import torch
import kmedoids as km
from scipy.spatial.distance import cdist

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from coreset.distance import extract_features, compute_distance_matrix, get_features_by_type

DATASET = 'SAN_BERNARDINO'
CFG_PATH = 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py'
RATIO = 0.1
SEED = 42

# Load dataset
print(f"Loading {DATASET}...")
cfg = import_config(CFG_PATH, verbose=False)
from basicts.data import TimeSeriesForecastingDataset
dataset = TimeSeriesForecastingDataset(mode='train', **cfg['DATASET']['PARAM'])
model_config = cfg['MODEL']

N = len(dataset)
k = int(N * RATIO)
print(f"N={N}, k={k}, ratio={RATIO}")

# Extract features and compute distance matrix (GPU)
print("Computing distance matrix...")
t0 = time.time()
inputs, targets = extract_features(dataset, model_config)
dist_np = compute_distance_matrix(inputs, targets, 'euclidean')
print(f"  Distance matrix: {time.time()-t0:.1f}s, shape={dist_np.shape}")

# Run FasterPAM (Rust)
print("\n=== FasterPAM (Rust, BUILD+SWAP) ===")
t0 = time.time()
result = km.fasterpam(dist_np, k, max_iter=100, init="build", random_state=SEED)
elapsed = time.time() - t0
improved_indices = list(result.medoids)
print(f"  Done: loss={result.loss:.2f}, n_iter={result.n_iter}, "
      f"n_swap={result.n_swap}, time={elapsed:.1f}s")

# Load old k-medoids and stride indices
old_path = f'coreset_indices/{DATASET}/k_medoids_euclidean_010_seed42.json'
with open(old_path) as f:
    old_indices = json.load(f)

stride_path = f'coreset_indices/{DATASET}/stride_euclidean_010_seed42.json'
with open(stride_path) as f:
    stride_indices = json.load(f)

# Compute costs on original distance matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
D = torch.from_numpy(dist_np).to(device).float()

improved_cost = D[:, improved_indices].min(dim=1).values.sum().item()
old_cost = D[:, old_indices].min(dim=1).values.sum().item()
stride_cost = D[:, stride_indices].min(dim=1).values.sum().item()

# Also compute QC in PCA-50 space (what proxy_metrics.json uses)
from sklearn.decomposition import PCA
raw_feat = get_features_by_type(inputs, targets, 'euclidean')
pca = PCA(n_components=50, random_state=42)
pca_feat = pca.fit_transform(raw_feat).astype(np.float32)

def qc_pca(indices):
    cs = pca_feat[indices]
    min_d = np.empty(len(pca_feat))
    for i in range(0, len(pca_feat), 500):
        end = min(i+500, len(pca_feat))
        min_d[i:end] = cdist(pca_feat[i:end], cs).min(axis=1)
    return np.mean(min_d)

qc_improved = qc_pca(improved_indices)
qc_old = qc_pca(old_indices)
qc_stride = qc_pca(stride_indices)

print(f"\n{'='*60}")
print(f"COMPARISON (ratio={RATIO}, seed={SEED}, {DATASET})")
print(f"{'='*60}")
print(f"{'Method':<25s} {'Raw L2 cost':>15s} {'PCA-50 QC':>15s}")
print(f"{'-'*55}")
print(f"{'FasterPAM (new)':<25s} {improved_cost:>15.2f} {qc_improved:>15.3f}")
print(f"{'Old k-med (5 iter)':<25s} {old_cost:>15.2f} {qc_old:>15.3f}")
print(f"{'Stride':<25s} {stride_cost:>15.2f} {qc_stride:>15.3f}")
print(f"\nFasterPAM vs stride: {(qc_stride - qc_improved)/qc_stride*100:+.1f}% (QC)")
print(f"FasterPAM vs old:    {(qc_old - qc_improved)/qc_old*100:+.1f}% (QC)")

# Save improved indices
out_path = f'coreset_indices/{DATASET}/k_medoids_v2_euclidean_010_seed42.json'
with open(out_path, 'w') as f:
    json.dump(improved_indices, f)
print(f"\nSaved to {out_path}")
