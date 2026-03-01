#!/usr/bin/env python
"""
Compute CPU-computable bound-related metrics for the generalization bound analysis.

1. Collect test MAE from checkpoint test_metrics.json
2. Quantization cost: (1/n) Σ min_{s∈S} d(x_i, s)  — k-medoids objective
3. W₁(P_test, P_train) approximation in PCA space
4. Merge with existing proxy metrics (sinkhorn_pca ≈ W₁(P_train, P_core))

Usage:
    conda activate cuda && python scripts/analysis/compute_bound_terms_cpu.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from basicts.data import TimeSeriesForecastingDataset

CKPT_BASE = Path('checkpoints/phase_b_deterministic')
OUTDIR = Path('experiments/result/analysis')


# ── Step 1: Collect test MAE from checkpoints ──────────────────────────────

def collect_test_mae() -> pd.DataFrame:
    """Scan all checkpoint directories and collect test MAE."""
    rows = []
    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        base = CKPT_BASE / model_name / 'xtraffic' / 'SAN_BERNARDINO_100_12_12' / '1'
        if not base.exists():
            continue
        for hash_dir in sorted(base.iterdir()):
            if not hash_dir.is_dir():
                continue
            cfg_file = hash_dir / 'cfg.txt'
            metrics_file = hash_dir / 'test_metrics.json'
            if not cfg_file.exists() or not metrics_file.exists():
                continue

            # Parse coreset config
            info = {}
            with open(cfg_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('SELECTION_STRATEGY:'):
                        info['method'] = line.split(':')[1].strip()
                    elif line.startswith('SELECTION_RATIO:'):
                        info['ratio'] = float(line.split(':')[1].strip())
                    elif line.startswith('DISTANCE_TYPE:'):
                        info['distance'] = line.split(':')[1].strip()
                    elif line.startswith('SEED:') and 'seed' not in info:
                        info['seed'] = int(line.split(':')[1].strip())

            # Read test MAE
            with open(metrics_file) as f:
                metrics = json.load(f)
            mae_test = metrics.get('overall', {}).get('MAE', None)

            # Read coreset size
            coreset_file = hash_dir / 'coreset-selection.json'
            n_coreset = None
            if coreset_file.exists():
                with open(coreset_file) as f:
                    n_coreset = len(json.load(f))

            rows.append({
                'model': model_name,
                **info,
                'mae_test': mae_test,
                'n_coreset': n_coreset,
                'ckpt_dir': str(hash_dir),
            })

    return pd.DataFrame(rows)


# ── Step 2: Compute quantization cost ──────────────────────────────────────

def extract_features_for_split(cfg_path: str, mode: str) -> np.ndarray:
    """Extract and flatten features for a dataset split."""
    from coreset.distance import extract_features, get_flat_features

    cfg = import_config(cfg_path, verbose=False)
    dataset = TimeSeriesForecastingDataset(mode=mode, **cfg['DATASET']['PARAM'])
    model_config = cfg['MODEL']

    inputs, targets = extract_features(dataset, model_config)
    flat = get_flat_features(inputs, targets)
    return flat


def compute_quantization_cost(train_features: np.ndarray,
                              coreset_indices: list,
                              batch_size: int = 500) -> dict:
    """Compute (1/n) Σ min_{s∈S} d(x_i, s) — the k-medoids objective.

    This is a direct proxy for W₁(P_train, P_core).
    """
    coreset_features = train_features[coreset_indices]
    n_train = train_features.shape[0]

    # Batched computation to avoid memory issues
    min_dists = np.empty(n_train, dtype=np.float64)
    for i in range(0, n_train, batch_size):
        end = min(i + batch_size, n_train)
        dists = cdist(train_features[i:end], coreset_features, metric='euclidean')
        min_dists[i:end] = dists.min(axis=1)

    return {
        'quantization_cost': float(np.mean(min_dists)),
        'quantization_max': float(np.max(min_dists)),
        'quantization_median': float(np.median(min_dists)),
        'quantization_std': float(np.std(min_dists)),
    }


def compute_quantization_pca(train_pca: np.ndarray,
                             coreset_indices: list,
                             batch_size: int = 1000) -> dict:
    """Quantization cost in PCA space."""
    coreset_pca = train_pca[coreset_indices]
    n_train = train_pca.shape[0]

    min_dists = np.empty(n_train, dtype=np.float64)
    for i in range(0, n_train, batch_size):
        end = min(i + batch_size, n_train)
        dists = cdist(train_pca[i:end], coreset_pca, metric='euclidean')
        min_dists[i:end] = dists.min(axis=1)

    return {
        'quant_pca_mean': float(np.mean(min_dists)),
        'quant_pca_max': float(np.max(min_dists)),
        'quant_pca_median': float(np.median(min_dists)),
    }


# ── Step 3: W₁(P_test, P_train) in PCA space ──────────────────────────────

def compute_w1_test_train_pca(train_pca: np.ndarray, test_pca: np.ndarray,
                              n_subsample: int = 3000) -> dict:
    """Approximate W₁(P_test, P_train) via average nearest-neighbor distance in PCA space.

    For each test point, find nearest train point. Mean of these distances ≈ W₁.
    This is an upper bound on W₁ (by the definition of transport cost).
    """
    rng = np.random.RandomState(42)

    # Subsample if needed
    if len(test_pca) > n_subsample:
        idx = rng.choice(len(test_pca), n_subsample, replace=False)
        test_sub = test_pca[idx]
    else:
        test_sub = test_pca

    if len(train_pca) > n_subsample:
        idx = rng.choice(len(train_pca), n_subsample, replace=False)
        train_sub = train_pca[idx]
    else:
        train_sub = train_pca

    # test → train: nearest neighbor distances
    min_dists_t2r = []
    for i in range(0, len(test_sub), 500):
        end = min(i + 500, len(test_sub))
        dists = cdist(test_sub[i:end], train_sub, metric='euclidean')
        min_dists_t2r.append(dists.min(axis=1))
    min_dists_t2r = np.concatenate(min_dists_t2r)

    # train → test: nearest neighbor distances
    min_dists_r2t = []
    for i in range(0, len(train_sub), 500):
        end = min(i + 500, len(train_sub))
        dists = cdist(train_sub[i:end], test_sub, metric='euclidean')
        min_dists_r2t.append(dists.min(axis=1))
    min_dists_r2t = np.concatenate(min_dists_r2t)

    return {
        'w1_test_train_pca': float(0.5 * np.mean(min_dists_t2r) + 0.5 * np.mean(min_dists_r2t)),
        'nn_test2train_mean': float(np.mean(min_dists_t2r)),
        'nn_train2test_mean': float(np.mean(min_dists_r2t)),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Collect test MAE ---
    print("Step 1: Collecting test MAE from checkpoints...")
    mae_df = collect_test_mae()
    smart = mae_df[mae_df['method'].isin(['k_medoids', 'k_center', 'graph_cut'])]
    print(f"  Total: {len(mae_df)} checkpoints, Smart: {len(smart)}")
    print(f"  Models: {smart['model'].value_counts().to_dict()}")

    # --- Step 2: Extract features (once) ---
    print("\nStep 2: Extracting features...")
    cfg_path = 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py'
    cfg = import_config(cfg_path, verbose=False)

    t0 = time.time()
    print("  Loading train features...", end=" ", flush=True)
    train_flat = extract_features_for_split(cfg_path, 'train')
    n_train = train_flat.shape[0]
    print(f"{n_train} samples, dim={train_flat.shape[1]} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  Loading test features...", end=" ", flush=True)
    test_flat = extract_features_for_split(cfg_path, 'test')
    n_test = test_flat.shape[0]
    print(f"{n_test} samples ({time.time()-t0:.1f}s)")

    # PCA reduction
    print("  Fitting PCA (dim=10)...", end=" ", flush=True)
    pca = PCA(n_components=10, random_state=42)
    train_pca = pca.fit_transform(train_flat)
    test_pca = pca.transform(test_flat)
    print(f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # --- Step 3: W₁(P_test, P_train) ---
    print("\nStep 3: Computing W₁(P_test, P_train) in PCA space...")
    w1_tt = compute_w1_test_train_pca(train_pca, test_pca)
    print(f"  W₁(test, train) ≈ {w1_tt['w1_test_train_pca']:.4f}")
    print(f"  NN test→train: {w1_tt['nn_test2train_mean']:.4f}")
    print(f"  NN train→test: {w1_tt['nn_train2test_mean']:.4f}")

    # --- Step 4: Per-coreset quantization cost ---
    print("\nStep 4: Computing quantization cost per coreset...")

    # Load coreset indices from index files
    quant_rows = []
    index_base = Path('coreset_indices/SAN_BERNARDINO')
    index_files = sorted(index_base.glob('*.json')) if index_base.exists() else []
    index_files = [f for f in index_files if f.name != 'proxy_metrics.json']
    print(f"  Found {len(index_files)} index files")

    for idx_file in index_files:
        # Parse filename: {method}_{distance}_{ratio}_seed{seed}.json
        name = idx_file.stem
        import re
        m = re.match(r'^(.+?)_(euclidean|temporal|spatial|combined)_(\d+)_seed(\d+)$', name)
        if not m:
            continue
        method, distance, ratio_pct, seed = m.groups()
        ratio = int(ratio_pct) / 100.0

        with open(idx_file) as f:
            indices = json.load(f)

        # Quantization cost in PCA space
        qc = compute_quantization_pca(train_pca, indices)

        quant_rows.append({
            'method': method,
            'distance': distance,
            'ratio': round(ratio, 4),
            'seed': int(seed),
            'n_coreset': len(indices),
            **qc,
        })

    quant_df = pd.DataFrame(quant_rows)
    if len(quant_df) > 0:
        quant_df = quant_df[quant_df['method'].isin(['k_medoids', 'k_center', 'graph_cut'])]
    print(f"  Computed for {len(quant_df)} smart coresets")

    # --- Step 5: Merge everything ---
    print("\nStep 5: Merging all data...")

    # Merge quantization with test MAE
    keys = ['method', 'distance', 'ratio', 'seed']

    # Load existing proxy metrics
    existing_pca = None
    pca_path = OUTDIR / 'pca_space_metrics.csv'
    if pca_path.exists():
        existing_pca = pd.read_csv(pca_path)
        existing_pca['ratio'] = existing_pca['ratio'].round(4)
        print(f"  Loaded existing PCA metrics: {len(existing_pca)} rows")

    # Build per-model result tables
    results_all = []
    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        model_mae = smart[smart['model'] == model_name].copy()
        if len(model_mae) == 0:
            continue

        # Merge with quantization (skip if empty)
        if len(quant_df) > 0:
            merged = model_mae.merge(quant_df, on=keys, how='left', suffixes=('', '_q'))
        else:
            merged = model_mae.copy()

        # Merge with existing sinkhorn_pca (≈ W₁(train, core))
        if existing_pca is not None:
            sinkhorn_cols = ['method', 'distance', 'ratio', 'seed',
                           'sinkhorn_pca', 'ot_pca']
            available = [c for c in sinkhorn_cols if c in existing_pca.columns]
            merged = merged.merge(existing_pca[available], on=keys, how='left')

        # Add W₁(test, train) as constant column
        merged['w1_test_train_pca'] = w1_tt['w1_test_train_pca']

        results_all.append(merged)

    result_df = pd.concat(results_all, ignore_index=True)

    # Save
    out_path = OUTDIR / 'bound_terms_cpu.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(result_df)} rows)")

    # --- Step 6: Summary ---
    print(f"\n{'='*90}")
    print("BOUND TERM ANALYSIS (CPU-computable components)")
    print(f"{'='*90}")

    print(f"\n  W₁(P_test, P_train) in PCA-10 space = {w1_tt['w1_test_train_pca']:.4f}")
    print(f"  (This is FIXED across all coreset settings — the test-train gap)")

    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        mdf = result_df[result_df['model'] == model_name]
        if len(mdf) == 0:
            continue
        model_short = 'STGCN' if 'STGCN' in model_name else model_name

        print(f"\n  {'='*80}")
        print(f"  Model: {model_short}")
        print(f"  {'='*80}")

        for ratio in [0.3, 0.7]:
            rdf = mdf[mdf['ratio'] == ratio]
            if len(rdf) == 0:
                continue

            print(f"\n  --- Ratio = {ratio} ---")
            header = f"  {'method':12s} {'distance':10s} {'MAE_test':>10s} {'quant_pca':>12s}"
            if 'sinkhorn_pca' in rdf.columns:
                header += f" {'sinkhorn_pca':>13s}"
            print(header)
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12}" +
                  (f" {'-'*13}" if 'sinkhorn_pca' in rdf.columns else ""))

            # Sort by MAE
            for _, row in rdf.sort_values('mae_test').iterrows():
                line = (f"  {row['method']:12s} {row['distance']:10s} "
                        f"{row['mae_test']:10.4f} {row.get('quant_pca_mean', float('nan')):12.4f}")
                if 'sinkhorn_pca' in rdf.columns:
                    line += f" {row.get('sinkhorn_pca', float('nan')):13.6f}"
                print(line)

    # --- Step 7: Correlation: quantization_cost vs MAE ---
    print(f"\n{'='*90}")
    print("CORRELATION: Quantization Cost vs MAE (within-setting)")
    print(f"{'='*90}")

    from scipy import stats
    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        mdf = result_df[result_df['model'] == model_name]
        model_short = 'STGCN' if 'STGCN' in model_name else model_name
        print(f"\n  {model_short}:")

        for ratio in [0.3, 0.7]:
            for dist in ['euclidean', 'temporal', 'spatial', 'combined']:
                sub = mdf[(mdf['ratio'] == ratio) & (mdf['distance'] == dist)]
                valid = sub[['quant_pca_mean', 'mae_test']].dropna()
                if len(valid) >= 4:
                    r, p = stats.pearsonr(valid['quant_pca_mean'], valid['mae_test'])
                    print(f"    ratio={ratio}, dist={dist:10s}: "
                          f"Pearson r={r:+.4f} (p={p:.3f}, n={len(valid)})")

    # Global correlation
    print(f"\n  Global (all settings pooled):")
    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        mdf = result_df[result_df['model'] == model_name]
        model_short = 'STGCN' if 'STGCN' in model_name else model_name
        valid = mdf[['quant_pca_mean', 'mae_test']].dropna()
        if len(valid) >= 4:
            r_p, p_p = stats.pearsonr(valid['quant_pca_mean'], valid['mae_test'])
            r_s, p_s = stats.spearmanr(valid['quant_pca_mean'], valid['mae_test'])
            print(f"    {model_short}: Pearson r={r_p:+.4f} (p={p_p:.4f}), "
                  f"Spearman ρ={r_s:+.4f} (p={p_s:.4f}), n={len(valid)}")

    if 'sinkhorn_pca' in result_df.columns:
        print(f"\n  Sinkhorn_pca vs MAE (global):")
        for model_name in ['STGCNChebGraphConv', 'AGCRN']:
            mdf = result_df[result_df['model'] == model_name]
            model_short = 'STGCN' if 'STGCN' in model_name else model_name
            valid = mdf[['sinkhorn_pca', 'mae_test']].dropna()
            if len(valid) >= 4:
                r_p, p_p = stats.pearsonr(valid['sinkhorn_pca'], valid['mae_test'])
                print(f"    {model_short}: Pearson r={r_p:+.4f} (p={p_p:.4f}), n={len(valid)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
