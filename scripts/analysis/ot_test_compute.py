import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.decomposition import PCA
import json, sys, os, re, time, torch

sys.path.append('/home/uqtyu7/github/BasicTS')
os.chdir('/home/uqtyu7/github/BasicTS')

from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type
from experiments.select_coreset import get_dataset_from_config
from coreset.ot_distance import _sinkhorn_cost

cfg = import_config('baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py', verbose=False)

# Train dataset
train_ds = get_dataset_from_config(cfg)
train_inputs, train_targets = extract_features(train_ds, cfg['MODEL'])
from coreset.distance import get_temporal_features, get_spatial_features

# Use raw flat features (no per-split normalization issues)
train_flat = np.concatenate([train_inputs.reshape(len(train_inputs), -1),
                              train_targets.reshape(len(train_inputs), -1)], axis=1)

# Test dataset
from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
test_ds = TimeSeriesForecastingDataset(
    dataset_name='xtraffic/SAN_BERNARDINO',
    train_val_test_ratio=[0.6, 0.2, 0.2],
    input_len=12, output_len=12,
    data_range=(0, 24192),
    mode='test'
)
test_inputs, test_targets = extract_features(test_ds, cfg['MODEL'])
test_flat = np.concatenate([test_inputs.reshape(len(test_inputs), -1),
                             test_targets.reshape(len(test_inputs), -1)], axis=1)
print(f"Train: {len(train_flat)}, Test: {len(test_flat)}")

# PCA on train, transform both, then standardize
pca = PCA(n_components=10, random_state=42)
train_pca_raw = pca.fit_transform(train_flat)
test_pca_raw = pca.transform(test_flat)

# Standardize using train stats so cost matrix has reasonable scale
pca_mean = train_pca_raw.mean(axis=0)
pca_std = train_pca_raw.std(axis=0)
pca_std[pca_std == 0] = 1.0
train_pca = ((train_pca_raw - pca_mean) / pca_std).astype(np.float32)
test_pca = ((test_pca_raw - pca_mean) / pca_std).astype(np.float32)
print(f"  PCA-10 standardized. Train std={train_pca.std():.3f}, Test std={test_pca.std():.3f}")

# Load index files
index_dir = Path('coreset_indices/SAN_BERNARDINO')
pattern = re.compile(r'^(.+?)_(euclidean|temporal|spatial|combined)_(\d+)_seed(\d+)$')
index_files = [f for f in sorted(index_dir.glob('*.json'))
               if f.name != 'proxy_metrics.json' and 'cosine' not in f.name]

# Subsample test
rng = np.random.RandomState(42)
n_sub = min(3000, len(test_pca))
test_sub = test_pca[rng.choice(len(test_pca), n_sub, replace=False)]
Y_test = torch.from_numpy(test_sub).cuda().float()

# Determine epsilon from median cost (adaptive)
from coreset.ot_distance import compute_cost_matrix
rng2 = np.random.RandomState(123)
X_sample = torch.from_numpy(train_pca[rng2.choice(len(train_pca), 500, replace=False)]).cuda().float()
Y_sample = torch.from_numpy(test_sub[rng2.choice(len(test_sub), 500, replace=False)]).cuda().float()
C_sample = compute_cost_matrix(X_sample, Y_sample)
print(f"  C_sample stats: min={C_sample.min():.4f} median={C_sample.median():.4f} max={C_sample.max():.4f}")
print(f"  X_sample stats: min={X_sample.min():.4f} mean={X_sample.mean():.4f} max={X_sample.max():.4f}")
print(f"  Y_sample stats: min={Y_sample.min():.4f} mean={Y_sample.mean():.4f} max={Y_sample.max():.4f}")
epsilon = float(C_sample.median().item() / 10)
max_iter = 100
print(f"  Adaptive epsilon = median(C)/10 = {epsilon:.4f}")

print("Computing OT(test, test)...")
ot_yy = _sinkhorn_cost(Y_test, Y_test, epsilon, max_iter)
print(f"  OT(Y,Y) = {ot_yy:.6f}")

rows = []
total = len(index_files)
for idx, f in enumerate(index_files):
    m = pattern.match(f.stem)
    if not m: continue
    method, distance, ratio_int, seed = m.groups()
    ratio = int(ratio_int) / 100.0
    seed_val = int(seed)

    with open(f) as fp:
        indices = json.load(fp)

    coreset_pca = train_pca[indices]
    X = torch.from_numpy(coreset_pca).cuda().float()

    ot_test = _sinkhorn_cost(X, Y_test, epsilon, max_iter)
    ot_xx = _sinkhorn_cost(X, X, epsilon, max_iter)
    sinkhorn_test = max(0.0, ot_test - 0.5 * ot_xx - 0.5 * ot_yy)

    rows.append({
        'method': method, 'distance': distance,
        'ratio': round(ratio, 4), 'seed': seed_val,
        'ot_test': round(ot_test, 6),
        'sinkhorn_test': round(sinkhorn_test, 6),
    })
    if (idx+1) % 10 == 0:
        print(f"  [{idx+1}/{total}] {f.stem}: OT={ot_test:.4f} Sink={sinkhorn_test:.6f}")

ot_df = pd.DataFrame(rows)
print(f'\nComputed OT(coreset, test) for {len(ot_df)} coresets')

# Merge with MAE
mae_df = pd.read_csv('experiments/result/analysis/full_metrics_with_mae.csv')
mae_df['ratio'] = mae_df['ratio'].round(4)
merged = ot_df.merge(mae_df[['method','distance','ratio','seed','MAE_mean','redundancy']],
                      on=['method','distance','ratio','seed'], how='inner')
merged = merged.dropna(subset=['MAE_mean'])

# Also get OT(coreset, train)
pca_df = pd.read_csv('experiments/result/analysis/pca_space_metrics.csv')
pca_df['ratio'] = pca_df['ratio'].round(4)
merged = merged.merge(pca_df[['method','distance','ratio','seed','ot_pca','sinkhorn_pca']],
                       on=['method','distance','ratio','seed'], how='left')

print(f'Valid rows: {len(merged)}')

print(f'\n{"="*60}')
print(f'=== Spearman rho with MAE ===')
print(f'{"="*60}')
for col in ['sinkhorn_test', 'ot_test', 'sinkhorn_pca', 'ot_pca', 'redundancy']:
    valid = merged[[col, 'MAE_mean']].dropna()
    if len(valid) < 5: continue
    r, p = stats.spearmanr(valid[col], valid['MAE_mean'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    label = col
    if col == 'sinkhorn_pca': label = 'sinkhorn_train (기존)'
    if col == 'ot_pca': label = 'ot_train (기존)'
    print(f'  {label:25s}  rho={r:+.4f}  p={p:.4f} {sig}')

print(f'\n=== Method ranking (ratio=0.3) ===')
r03 = merged[merged['ratio'] == 0.3].groupby('method').agg(
    sink_test=('sinkhorn_test','mean'),
    sink_train=('sinkhorn_pca','mean'),
    MAE=('MAE_mean','mean')
).sort_values('MAE')
print(f'{"method":12s} {"sink_test":>12s} {"sink_train":>12s} {"MAE":>8s}')
for m, row in r03.iterrows():
    print(f'{m:12s} {row["sink_test"]:12.6f} {row["sink_train"]:12.6f} {row["MAE"]:8.4f}')

# Save results
ot_df.to_csv('experiments/result/analysis/ot_test_metrics.csv', index=False)
print(f'\nSaved to experiments/result/analysis/ot_test_metrics.csv')
