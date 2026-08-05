#!/usr/bin/env python
"""
Coreset Advantage Analysis:
"When and why does coreset outperform full data?"

Mathematical formulation:
  Full data:  E_test(f_full) ≈ Ê_train(f_full) + L · W₁(P_test, P_train)
  Coreset:    E_test(f_S)    ≈ Ê_S(f_S)        + L · W₁(P_test, P_S)

  Coreset outperforms when W₁(P_test, P_S) < W₁(P_test, P_train),
  i.e., the coreset is distributionally closer to the test set than the
  full training data — possible when temporal distribution shift exists.

Computes:
  1. SW₁(P_test, P_train) — baseline (fixed)
  2. SW₁(P_test, P_core) — per coreset
  3. SW₁(P_core, P_train) — per coreset (train coverage)
  4. ΔW₁ = SW₁(test,train) - SW₁(test,core)  (positive = coreset advantage)

All distances use Sliced Wasserstein in PCA-10 feature space.

Usage:
    conda activate cuda && python scripts/analysis/coreset_advantage_analysis.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from basicts.data import TimeSeriesForecastingDataset

CKPT_BASE = Path('checkpoints/phase_b_deterministic')
OUTDIR = Path('experiments/result/analysis')


# ── Feature Extraction ───────────────────────────────────────────────────────

def extract_features_for_split(cfg_path: str, mode: str) -> np.ndarray:
    from coreset.distance import extract_features, get_flat_features
    cfg = import_config(cfg_path, verbose=False)
    dataset = TimeSeriesForecastingDataset(mode=mode, **cfg['DATASET']['PARAM'])
    model_config = cfg['MODEL']
    inputs, targets = extract_features(dataset, model_config)
    return get_flat_features(inputs, targets)


# ── Sliced Wasserstein Distance ───────────────────────────────────────────────

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray,
                       n_projections: int = 500, seed: int = 42) -> float:
    """Sliced Wasserstein distance between two empirical distributions.

    SW₁(P, Q) = E_θ[ W₁(θ#P, θ#Q) ]

    1D W₁ is computed via quantile matching (sorting).
    Properly handles different sample sizes via interpolation.
    """
    rng = np.random.RandomState(seed)
    d = X.shape[1]

    # Random unit directions on d-sphere
    directions = rng.randn(n_projections, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Project
    proj_x = X @ directions.T  # (n, n_proj)
    proj_y = Y @ directions.T  # (m, n_proj)

    n_quantiles = max(len(X), len(Y))
    qs = np.linspace(0, 1, n_quantiles, endpoint=True)

    w1_sum = 0.0
    for j in range(n_projections):
        sx = np.sort(proj_x[:, j])
        sy = np.sort(proj_y[:, j])

        # Interpolate to uniform quantile grid
        qx = np.interp(qs, np.linspace(0, 1, len(sx)), sx)
        qy = np.interp(qs, np.linspace(0, 1, len(sy)), sy)

        w1_sum += np.mean(np.abs(qx - qy))

    return float(w1_sum / n_projections)


# ── Collect Test MAE ──────────────────────────────────────────────────────────

def collect_test_mae() -> pd.DataFrame:
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

            with open(metrics_file) as f:
                metrics = json.load(f)
            mae_test = metrics.get('overall', {}).get('MAE', None)

            rows.append({
                'model': model_name, **info,
                'mae_test': mae_test,
                'ckpt_dir': str(hash_dir),
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect test MAE ──
    print("Step 1: Collecting test MAE...")
    mae_df = collect_test_mae()
    smart = mae_df[mae_df['method'].isin(['k_medoids', 'k_center', 'graph_cut'])]
    print(f"  {len(smart)} smart checkpoints ({smart['model'].value_counts().to_dict()})")

    # ── 2. Feature extraction & PCA ──
    print("\nStep 2: Feature extraction & PCA...")
    cfg_path = 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py'

    train_flat = extract_features_for_split(cfg_path, 'train')
    test_flat = extract_features_for_split(cfg_path, 'test')
    print(f"  Train: {train_flat.shape}, Test: {test_flat.shape}")

    pca = PCA(n_components=10, random_state=42)
    train_pca = pca.fit_transform(train_flat)
    test_pca = pca.transform(test_flat)
    ev = pca.explained_variance_ratio_.sum()
    print(f"  PCA-10 explained variance: {ev:.3f}")

    # ── 3. Baseline: SW₁(test, train) ──
    print("\nStep 3: SW₁(P_test, P_train)...")
    t0 = time.time()
    sw1_test_train = sliced_wasserstein(test_pca, train_pca)
    print(f"  SW₁(test, train) = {sw1_test_train:.4f}  ({time.time()-t0:.1f}s)")

    # ── 4. Per-coreset: SW₁(test, core) and SW₁(core, train) ──
    print("\nStep 4: Per-coreset Sliced Wasserstein distances...")
    index_base = Path('coreset_indices/SAN_BERNARDINO')
    index_files = sorted(index_base.glob('*.json'))
    index_files = [f for f in index_files if f.name != 'proxy_metrics.json']

    sw1_rows = []
    for idx_file in index_files:
        name = idx_file.stem
        m = re.match(r'^(.+?)_(euclidean|temporal|spatial|combined)_(\d+)_seed(\d+)$', name)
        if not m:
            continue
        method, distance, ratio_pct, seed_str = m.groups()
        # Phase B Euclidean K-medoids checkpoints used the original selector.
        # Canonical k_medoids_euclidean_* files were later replaced by the
        # Phase D PCA/L1 selector, while the original files were preserved as
        # k_medoids_old_euclidean_*.  Map the preserved files back to the
        # checkpoint method label and skip the overwritten canonical copies.
        if method == 'k_medoids' and distance == 'euclidean':
            continue
        if method == 'k_medoids_old' and distance == 'euclidean':
            method = 'k_medoids'
        if method not in ('k_medoids', 'k_center', 'graph_cut'):
            continue
        ratio = int(ratio_pct) / 100.0

        with open(idx_file) as f:
            indices = json.load(f)

        core_pca = train_pca[indices]

        t0 = time.time()
        sw1_tc = sliced_wasserstein(test_pca, core_pca)     # W₁(test, core)
        sw1_ct = sliced_wasserstein(core_pca, train_pca)    # W₁(core, train)
        dt = time.time() - t0

        delta_w1 = sw1_test_train - sw1_tc  # positive = coreset closer to test

        sw1_rows.append({
            'method': method,
            'distance': distance,
            'ratio': round(ratio, 4),
            'seed': int(seed_str),
            'sw1_test_core': sw1_tc,
            'sw1_core_train': sw1_ct,
            'sw1_test_train': sw1_test_train,
            'delta_w1': delta_w1,
            'n_coreset': len(indices),
        })

        sign = '+' if delta_w1 > 0 else ' '
        print(f"  {method:12s} {distance:10s} r={ratio:.1f} s={seed_str}: "
              f"SW₁(t,c)={sw1_tc:.4f}  SW₁(c,tr)={sw1_ct:.4f}  "
              f"ΔW₁={sign}{delta_w1:.4f}  ({dt:.1f}s)")

    sw1_df = pd.DataFrame(sw1_rows)

    # ── 5. Merge & Analyze ──
    print(f"\n{'='*90}")
    print("CORESET ADVANTAGE ANALYSIS")
    print(f"{'='*90}")
    print(f"\nBaseline: SW₁(P_test, P_train) = {sw1_test_train:.4f}")
    print(f"If SW₁(P_test, P_core) < {sw1_test_train:.4f}, coreset is closer to test.\n")

    keys = ['method', 'distance', 'ratio', 'seed']
    all_merged = []

    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        model_mae = smart[smart['model'] == model_name].copy()
        if len(model_mae) == 0:
            continue
        model_short = 'STGCN' if 'STGCN' in model_name else model_name

        merged = model_mae.merge(sw1_df, on=keys, how='inner')
        all_merged.append(merged)

        print(f"{'='*80}")
        print(f"  Model: {model_short} (n={len(merged)})")
        print(f"{'='*80}")

        # Count advantage cases
        closer = merged[merged['delta_w1'] > 0]
        farther = merged[merged['delta_w1'] <= 0]
        print(f"\n  Coreset CLOSER to test (ΔW₁ > 0): {len(closer)}/{len(merged)}")
        if len(closer) > 0:
            print(f"    Methods: {closer['method'].value_counts().to_dict()}")
            print(f"    Ratios:  {closer['ratio'].value_counts().to_dict()}")
            print(f"    MAE: {closer['mae_test'].mean():.4f} ± {closer['mae_test'].std():.4f}")

        print(f"  Coreset FARTHER from test (ΔW₁ ≤ 0): {len(farther)}/{len(merged)}")
        if len(farther) > 0:
            print(f"    MAE: {farther['mae_test'].mean():.4f} ± {farther['mae_test'].std():.4f}")

        # Global correlations
        print(f"\n  Correlations (global, n={len(merged)}):")
        for xvar, xlabel in [('sw1_test_core', 'SW₁(test,core)'),
                              ('delta_w1', 'ΔW₁'),
                              ('sw1_core_train', 'SW₁(core,train)')]:
            valid = merged[[xvar, 'mae_test']].dropna()
            if len(valid) >= 4:
                r, p = stats.pearsonr(valid[xvar], valid['mae_test'])
                rho, ps = stats.spearmanr(valid[xvar], valid['mae_test'])
                print(f"    {xlabel:20s} vs MAE: r={r:+.4f} (p={p:.4f}), ρ={rho:+.4f}")

        # Within-setting correlations (fixed ratio)
        print(f"\n  Within-setting correlations:")
        for ratio in [0.3, 0.7]:
            rdf = merged[merged['ratio'] == ratio]
            valid = rdf[['sw1_test_core', 'mae_test']].dropna()
            if len(valid) >= 4:
                r, p = stats.pearsonr(valid['sw1_test_core'], valid['mae_test'])
                rho, _ = stats.spearmanr(valid['sw1_test_core'], valid['mae_test'])
                print(f"    ratio={ratio}: SW₁(test,core) vs MAE: "
                      f"r={r:+.4f}, ρ={rho:+.4f} (n={len(valid)})")

        # Detailed table
        print(f"\n  {'method':12s} {'dist':10s} {'r':>4s} {'s':>3s} "
              f"{'SW₁(t,c)':>10s} {'SW₁(c,tr)':>10s} {'ΔW₁':>8s} {'MAE':>10s}")
        print(f"  {'-'*12} {'-'*10} {'-'*4} {'-'*3} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
        for _, row in merged.sort_values('mae_test').iterrows():
            sign = '+' if row['delta_w1'] > 0 else ' '
            print(f"  {row['method']:12s} {row['distance']:10s} "
                  f"{row['ratio']:4.1f} {int(row['seed']):3d} "
                  f"{row['sw1_test_core']:10.4f} {row['sw1_core_train']:10.4f} "
                  f"{sign}{row['delta_w1']:7.4f} {row['mae_test']:10.4f}")
        print()

    result = pd.concat(all_merged, ignore_index=True) if all_merged else pd.DataFrame()

    # ── 6. Save CSV ──
    out_csv = OUTDIR / 'coreset_advantage.csv'
    result.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # ── 7. Visualization ──
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    method_colors = {'k_medoids': '#2196F3', 'k_center': '#FF9800', 'graph_cut': '#4CAF50'}
    ratio_markers = {0.3: 'o', 0.7: 's'}

    for col, model_name in enumerate(['STGCNChebGraphConv', 'AGCRN']):
        mdf = result[result['model'] == model_name]
        if len(mdf) == 0:
            continue
        model_short = 'STGCN' if 'STGCN' in model_name else model_name

        # Row 0: SW₁(test, core) vs MAE
        ax = axes[0, col]
        for ratio, marker in ratio_markers.items():
            for method, color in method_colors.items():
                sub = mdf[(mdf['ratio'] == ratio) & (mdf['method'] == method)]
                ax.scatter(sub['sw1_test_core'], sub['mae_test'],
                           c=color, marker=marker, s=80, alpha=0.7,
                           label=f'{method} r={ratio}')

        # Baseline line
        ax.axvline(x=sw1_test_train, color='red', linestyle='--', alpha=0.6,
                   label=f'SW₁(test,train)={sw1_test_train:.2f}')

        ax.set_xlabel('SW₁(P_test, P_core)')
        ax.set_ylabel('Test MAE')
        ax.set_title(f'{model_short}: Test Proximity vs Performance')
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.annotate('← Closer to test | Farther from test →',
                    xy=(0.5, 0.02), xycoords='axes fraction',
                    ha='center', fontsize=8, color='gray')

        # Row 1: 2D view — SW₁(core,train) vs SW₁(test,core), color=MAE
        ax = axes[1, col]
        sc = ax.scatter(mdf['sw1_core_train'], mdf['sw1_test_core'],
                        c=mdf['mae_test'], cmap='RdYlGn_r', s=80, alpha=0.8,
                        edgecolors='gray', linewidths=0.5)

        # Annotate method for each point
        for _, row in mdf.iterrows():
            ax.annotate(row['method'][:3], (row['sw1_core_train'], row['sw1_test_core']),
                        fontsize=5, alpha=0.6, ha='center', va='bottom')

        ax.axhline(y=sw1_test_train, color='red', linestyle='--', alpha=0.4,
                   label=f'SW₁(test,train)')
        ax.set_xlabel('SW₁(P_core, P_train) — Train Coverage')
        ax.set_ylabel('SW₁(P_test, P_core) — Test Proximity')
        ax.set_title(f'{model_short}: Distributional Trade-off')
        plt.colorbar(sc, ax=ax, label='MAE', shrink=0.8)
        ax.legend(fontsize=7)
        ax.annotate('Ideal: bottom-left\n(close to both)',
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plot_path = OUTDIR / 'coreset_advantage.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")

    # ── 8. Summary ──
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"""
Mathematical Formulation:
  Full data:  E_test(f_full) ≈ Ê_train + L · W₁(P_test, P_train)
  Coreset:    E_test(f_S)    ≈ Ê_S     + L · W₁(P_test, P_S)

  Coreset advantage when:
    W₁(P_test, P_S) < W₁(P_test, P_train)
    i.e., coreset is distributionally closer to test than full train.

  This happens because:
    - Temporal distribution shift in traffic data (train ≠ test period)
    - Coreset removes training patterns far from test distribution
    - Uniform reweighting (1/|S|) shifts mass toward representative patterns

  The trade-off:
    W₁(P_core, P_train) ↑ (worse train coverage)
    W₁(P_test, P_core)  ↓ (better test coverage)

  Optimal coreset balances both terms.

Empirical Results (SW₁ in PCA-10 space):
  SW₁(P_test, P_train) = {sw1_test_train:.4f} (baseline)
""")

    for model_name in ['STGCNChebGraphConv', 'AGCRN']:
        mdf = result[result['model'] == model_name]
        if len(mdf) == 0:
            continue
        model_short = 'STGCN' if 'STGCN' in model_name else model_name
        n_closer = (mdf['delta_w1'] > 0).sum()
        print(f"  {model_short}: {n_closer}/{len(mdf)} coresets closer to test than full train")

    print("\nDone.")


if __name__ == '__main__':
    main()
