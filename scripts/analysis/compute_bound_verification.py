#!/usr/bin/env python
"""
Bound Verification Table for Coreset Forecasting.

Computes all terms for TWO theoretical bounds:

1) Single-model bound (no ε_disc):
   MAE_test(f_core) ≤ MAE^w_C(f_core) + L_core · [W₁(test,train) + W₁(train,C;w)]

2) Gap bound (for comparison):
   G_test ≤ (L_train + L_core) · [W₁(test,train) + W₁(train,C;w)] + ε_disc

Lipschitz estimation uses kNN-based approach (k=20 nearest neighbours per point)
for more accurate estimation than random pairs.

Usage (on GPU node):
    sgpu-cuda
    conda activate cuda && python scripts/analysis/compute_bound_verification.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from basicts.data import TimeSeriesForecastingDataset

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'STGCNChebGraphConv': {
        'cfg': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
        'full_ckpt': 'checkpoints/phase_b_full_data/STGCNChebGraphConv/xtraffic/'
                     'SAN_BERNARDINO_100_12_12/1/86896e08189d88aec79fd19a3b53f284/'
                     'STGCNChebGraphConv_best_val_MAE.pt',
    },
    'AGCRN': {
        'cfg': 'baselines/AGCRN/SAN_BERNARDINO/SAN_BERNARDINO.py',
        'full_ckpt': 'checkpoints/phase_b_full_data/AGCRN/xtraffic/'
                     'SAN_BERNARDINO_100_12_12/1/1fd65566c20c348d88d63f11d14a091d/'
                     'AGCRN_best_val_MAE.pt',
    },
}

CKPT_BASE = Path('checkpoints/phase_b_deterministic')


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def discover_coreset_checkpoints(model_name: str) -> list:
    base = CKPT_BASE / model_name / 'xtraffic' / 'SAN_BERNARDINO_100_12_12' / '1'
    if not base.exists():
        return []
    smart_methods = {'k_medoids', 'k_center', 'graph_cut'}
    results = []
    for hash_dir in sorted(base.iterdir()):
        if not hash_dir.is_dir():
            continue
        ckpt_file = hash_dir / f'{model_name}_best_val_MAE.pt'
        cfg_file = hash_dir / 'cfg.txt'
        coreset_file = hash_dir / 'coreset-selection.json'
        if not all(f.exists() for f in [ckpt_file, cfg_file, coreset_file]):
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
        if info.get('method') not in smart_methods:
            continue
        info['ckpt_path'] = str(ckpt_file)
        info['ckpt_dir'] = str(hash_dir)
        with open(coreset_file) as f:
            info['coreset_indices'] = json.load(f)
        results.append(info)
    return results


# ── Model helpers ────────────────────────────────────────────────────────────

def build_model_and_scaler(cfg_path: str, device: torch.device):
    cfg = import_config(cfg_path, verbose=False)
    model = cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM']).to(device)
    model.eval()
    scaler = cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
    return model, scaler, cfg


def load_weights(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()


# ── Compute per-sample MAE ───────────────────────────────────────────────────

@torch.no_grad()
def compute_per_sample_mae(model, scaler, cfg, dataloader, device) -> np.ndarray:
    """Return per-sample MAE as a 1-D numpy array (one scalar per sample)."""
    fwd_feat = cfg['MODEL'].get('FORWARD_FEATURES', None)
    tgt_feat = cfg['MODEL'].get('TARGET_FEATURES', None)
    null_val = cfg['METRICS'].get('NULL_VAL', np.nan)

    sample_maes = []
    for batch in dataloader:
        inputs = batch['inputs'].to(device).float()
        target = batch['target'].to(device).float()

        inp = scaler.transform(inputs) if scaler else inputs
        history = inp[..., fwd_feat] if fwd_feat is not None else inp

        pred = model(history_data=history, future_data=None,
                     batch_seen=None, epoch=None, train=False)
        if isinstance(pred, dict):
            pred = pred['prediction']
        if scaler is not None and scaler.rescale:
            pred = scaler.inverse_transform(pred)

        tgt = target[..., tgt_feat] if tgt_feat is not None else target

        # Per-sample masked MAE
        abs_err = torch.abs(pred - tgt)
        if np.isnan(null_val):
            mask = ~torch.isnan(tgt)
        else:
            mask = (tgt != null_val)
        dims = tuple(range(1, abs_err.ndim))
        counts = mask.float().sum(dim=dims).clamp(min=1)
        per_sample = (abs_err * mask.float()).sum(dim=dims) / counts
        sample_maes.append(per_sample.cpu().numpy())

    return np.concatenate(sample_maes)


# ── PCA embeddings ───────────────────────────────────────────────────────────

def compute_pca_embeddings(dataset, n_components=10, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_inputs = []
    for batch in loader:
        inp = batch['inputs'].float()
        all_inputs.append(inp.reshape(inp.shape[0], -1).numpy())
    X = np.concatenate(all_inputs, axis=0)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca


# ── Lipschitz estimation ────────────────────────────────────────────────────

def estimate_lipschitz(sample_mae: np.ndarray, pca_emb: np.ndarray,
                       n_pairs: int = 2000, knn_k: int = 20) -> dict:
    """Estimate L = max |φ_f(x_i) - φ_f(x_j)| / d_PCA(x_i, x_j).

    Uses kNN-based estimation: for each point, compute the ratio with its
    k nearest neighbours in PCA space. This is far more effective than
    random pairs at finding the worst-case Lipschitz ratio, because
    nearby points with different losses produce the largest ratios.
    """
    from scipy.spatial import KDTree

    n = len(sample_mae)

    # ── kNN-based estimation ──
    tree = KDTree(pca_emb)
    dists, indices = tree.query(pca_emb, k=knn_k + 1)  # +1 for self
    nn_dists = dists[:, 1:]      # (n, knn_k)
    nn_indices = indices[:, 1:]  # (n, knn_k)

    all_ratios = []
    for ki in range(knn_k):
        phi_diff = np.abs(sample_mae - sample_mae[nn_indices[:, ki]])
        d = nn_dists[:, ki]
        valid = d > 1e-8
        ratios = phi_diff[valid] / d[valid]
        all_ratios.append(ratios)
    all_ratios = np.concatenate(all_ratios)

    return {
        'L_max': float(np.max(all_ratios)) if len(all_ratios) > 0 else 0.0,
        'L_95': float(np.percentile(all_ratios, 95)) if len(all_ratios) > 0 else 0.0,
        'L_median': float(np.median(all_ratios)) if len(all_ratios) > 0 else 0.0,
        'n_pairs_checked': len(all_ratios),
    }


# ── Cluster weights ─────────────────────────────────────────────────────────

def compute_cluster_weights(train_pca: np.ndarray,
                            coreset_indices: list) -> np.ndarray:
    """Nearest-assignment cluster weights w_ℓ = n_ℓ / n."""
    coreset_pca = train_pca[coreset_indices]
    n = len(train_pca)
    k = len(coreset_indices)

    # Batch distance computation to avoid memory issues
    assignments = np.empty(n, dtype=int)
    batch = 2000
    for s in range(0, n, batch):
        e = min(s + batch, n)
        dists = cdist(train_pca[s:e], coreset_pca)
        assignments[s:e] = np.argmin(dists, axis=1)

    cluster_sizes = np.bincount(assignments, minlength=k)
    return cluster_sizes / n


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--models', nargs='+',
                        default=['STGCNChebGraphConv', 'AGCRN'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--knn-k', type=int, default=20,
                        help='k for kNN-based Lipschitz estimation')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CPU precomputed W₁ terms
    cpu_path = Path('experiments/result/analysis/bound_terms_cpu.csv')
    cpu_df = pd.read_csv(cpu_path) if cpu_path.exists() else None
    if cpu_df is not None:
        print(f"Loaded CPU terms: {len(cpu_df)} rows")

    outdir = Path('experiments/result/analysis')
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"WARNING: No config for {model_name}, skipping")
            continue

        mcfg = MODEL_CONFIGS[model_name]
        print(f"\n{'='*70}\n  Model: {model_name}\n{'='*70}")

        # Build model architecture, scaler, config
        model, scaler, cfg = build_model_and_scaler(mcfg['cfg'], device)
        null_val = cfg['METRICS'].get('NULL_VAL', np.nan)

        # Datasets
        ds_params = cfg['DATASET']['PARAM']
        train_ds = TimeSeriesForecastingDataset(mode='train', **ds_params)
        test_ds = TimeSeriesForecastingDataset(mode='test', **ds_params)
        n_train = len(train_ds)
        print(f"  Train: {n_train}, Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        # PCA embeddings (train set, for distances)
        print("  Computing PCA-10 embeddings on train set...")
        train_pca, _ = compute_pca_embeddings(train_ds, n_components=10)

        # ── Full-data model (f_train) ────────────────────────────────────
        print(f"  Loading full-data model...")
        load_weights(model, mcfg['full_ckpt'], device)

        # Per-sample MAE on train (for L_train + MAE^w later)
        print("  Computing per-sample MAE on train (f_train)...")
        sm_train_ftrain = compute_per_sample_mae(
            model, scaler, cfg, train_loader, device)

        # Per-sample MAE on test (for MAE_test(f_train))
        print("  Computing per-sample MAE on test (f_train)...")
        sm_test_ftrain = compute_per_sample_mae(
            model, scaler, cfg, test_loader, device)
        mae_test_ftrain = float(np.mean(sm_test_ftrain))
        mae_train_ftrain = float(np.mean(sm_train_ftrain))
        print(f"  MAE_test(f_train) = {mae_test_ftrain:.4f}")
        print(f"  MAE_train(f_train) = {mae_train_ftrain:.4f}")

        # L_train
        print("  Estimating L_train...")
        L_train = estimate_lipschitz(sm_train_ftrain, train_pca,
                                     knn_k=args.knn_k)
        print(f"  L_train: max={L_train['L_max']:.6f}, "
              f"95%={L_train['L_95']:.6f}, "
              f"median={L_train['L_median']:.6f}  "
              f"({L_train['n_pairs_checked']:,} pairs)")

        # ── Coreset checkpoints ──────────────────────────────────────────
        ckpts = discover_coreset_checkpoints(model_name)
        print(f"\n  Found {len(ckpts)} coreset checkpoints")

        for i, ck in enumerate(tqdm(ckpts, desc=f"  {model_name}")):
            method = ck['method']
            distance = ck['distance']
            ratio = ck['ratio']
            seed = ck['seed']
            coreset_idx = ck['coreset_indices']

            # Load coreset model
            load_weights(model, ck['ckpt_path'], device)

            # Per-sample MAE on train (f_core)
            sm_train_fcore = compute_per_sample_mae(
                model, scaler, cfg, train_loader, device)

            # Per-sample MAE on test (f_core)
            sm_test_fcore = compute_per_sample_mae(
                model, scaler, cfg, test_loader, device)
            mae_test_fcore = float(np.mean(sm_test_fcore))
            mae_train_fcore = float(np.mean(sm_train_fcore))

            # G_test (actual gap)
            G_test = abs(mae_test_ftrain - mae_test_fcore)

            # (A), (B), (C) decomposition
            term_A = abs(mae_test_ftrain - mae_train_ftrain)
            term_B = abs(mae_train_ftrain - mae_train_fcore)
            term_C = abs(mae_train_fcore - mae_test_fcore)

            # Cluster weights
            weights = compute_cluster_weights(train_pca, coreset_idx)

            # ε_disc = |MAE^w(f_train) - MAE^w(f_core)|
            mae_w_ftrain = float(np.sum(
                weights * sm_train_ftrain[coreset_idx]))
            mae_w_fcore = float(np.sum(
                weights * sm_train_fcore[coreset_idx]))
            eps_disc = abs(mae_w_ftrain - mae_w_fcore)

            # L_core
            L_core = estimate_lipschitz(sm_train_fcore, train_pca,
                                        knn_k=args.knn_k)

            # W₁ terms (from CPU precomputation)
            w1_tt = 13.02  # default
            w1_tc = None
            if cpu_df is not None:
                match = cpu_df[
                    (cpu_df['model'] == model_name) &
                    (cpu_df['method'] == method) &
                    (cpu_df['ratio'] == ratio) &
                    (cpu_df['distance'] == distance) &
                    (cpu_df['seed'] == seed)
                ]
                if len(match) == 1:
                    w1_tt = float(match['w1_test_train_pca'].values[0])
                    w1_tc = float(match['quant_pca_mean'].values[0])

            # Fallback: compute quantization cost directly
            if w1_tc is None:
                coreset_pca = train_pca[coreset_idx]
                dists = cdist(train_pca, coreset_pca)
                w1_tc = float(np.mean(np.min(dists, axis=1)))

            # ── (B) sub-decomposition: B1 + ε_disc + B3 ─────────────────
            # B1 = |MAE_train(f_train) - MAE^w_C(f_train)|
            # B3 = |MAE^w_C(f_core) - MAE_train(f_core)|
            term_B1 = abs(mae_train_ftrain - mae_w_ftrain)
            term_B3 = abs(mae_w_fcore - mae_train_fcore)

            # ── Assemble bound ───────────────────────────────────────────
            # Per-term bounds: (A) ≤ L_train·W₁_tt, (C) ≤ L_core·W₁_tt
            # B1 ≤ L_train·W₁_tc, B3 ≤ L_core·W₁_tc
            bound_A_95 = L_train['L_95'] * w1_tt
            bound_C_95 = L_core['L_95'] * w1_tt
            bound_B1_95 = L_train['L_95'] * w1_tc
            bound_B3_95 = L_core['L_95'] * w1_tc
            bound_B_95 = ((L_train['L_95'] + L_core['L_95'])
                          * w1_tc + eps_disc)
            bound_95 = ((L_train['L_95'] + L_core['L_95'])
                        * (w1_tt + w1_tc) + eps_disc)
            bound_max = ((L_train['L_max'] + L_core['L_max'])
                         * (w1_tt + w1_tc) + eps_disc)

            # ── Single-model bound ────────────────────────────────────
            # MAE_test(f_core) ≤ MAE^w_C(f_core) + L_core · [W₁_tt + W₁_tc]
            single_LW1_95 = L_core['L_95'] * (w1_tt + w1_tc)
            single_LW1_max = L_core['L_max'] * (w1_tt + w1_tc)
            single_bound_95 = mae_w_fcore + single_LW1_95
            single_bound_max = mae_w_fcore + single_LW1_max
            single_holds_95 = single_bound_95 >= mae_test_fcore - 1e-6
            single_holds_max = single_bound_max >= mae_test_fcore - 1e-6
            # How much of the generalization gap does L·W₁ cover?
            gen_gap = mae_test_fcore - mae_w_fcore
            single_coverage_95 = (single_LW1_95 / gen_gap
                                  if gen_gap > 1e-6 else float('inf'))

            result = {
                'model': model_name,
                'method': method,
                'distance': distance,
                'ratio': ratio,
                'seed': seed,
                # Actual
                'mae_test_ftrain': round(mae_test_ftrain, 4),
                'mae_test_fcore': round(mae_test_fcore, 4),
                'mae_train_ftrain': round(mae_train_ftrain, 4),
                'mae_train_fcore': round(mae_train_fcore, 4),
                'G_test': round(G_test, 4),
                # (A)(B)(C) decomposition
                'term_A': round(term_A, 4),
                'term_B': round(term_B, 4),
                'term_C': round(term_C, 4),
                # Bound terms
                'L_train_95': round(L_train['L_95'], 6),
                'L_train_max': round(L_train['L_max'], 6),
                'L_core_95': round(L_core['L_95'], 6),
                'L_core_max': round(L_core['L_max'], 6),
                'W1_test_train': round(w1_tt, 4),
                'W1_train_core': round(w1_tc, 4),
                'eps_disc': round(eps_disc, 4),
                # (B) sub-decomposition
                'term_B1': round(term_B1, 4),
                'term_B3': round(term_B3, 4),
                # Per-term bounds (L95)
                'bound_A_95': round(bound_A_95, 6),
                'bound_B1_95': round(bound_B1_95, 6),
                'bound_B3_95': round(bound_B3_95, 6),
                'bound_B_95': round(bound_B_95, 4),
                'bound_C_95': round(bound_C_95, 6),
                'mae_w_ftrain': round(mae_w_ftrain, 4),
                'mae_w_fcore': round(mae_w_fcore, 4),
                # Gap bound (two variants)
                'bound_L95': round(bound_95, 4),
                'bound_Lmax': round(bound_max, 4),
                'ratio_L95': (round(bound_95 / G_test, 2)
                              if G_test > 1e-6 else float('inf')),
                'ratio_Lmax': (round(bound_max / G_test, 2)
                               if G_test > 1e-6 else float('inf')),
                'holds_L95': bound_95 >= G_test - 1e-6,
                'holds_Lmax': bound_max >= G_test - 1e-6,
                # Single-model bound
                'single_bound_95': round(single_bound_95, 4),
                'single_bound_max': round(single_bound_max, 4),
                'single_LW1_95': round(single_LW1_95, 4),
                'single_gen_gap': round(gen_gap, 4),
                'single_coverage_95': round(single_coverage_95, 4),
                'single_holds_95': single_holds_95,
                'single_holds_max': single_holds_max,
            }
            all_results.append(result)

    # ── Save ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    out_path = outdir / 'bound_verification.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    if len(df) > 0:
        print(f"\n{'='*70}")
        print("BOUND VERIFICATION SUMMARY")
        print(f"{'='*70}")

        for L_key, bcol, rcol, hcol in [
            ('L_95', 'bound_L95', 'ratio_L95', 'holds_L95'),
            ('L_max', 'bound_Lmax', 'ratio_Lmax', 'holds_Lmax'),
        ]:
            holds = df[hcol].sum()
            total = len(df)
            print(f"\n  [{L_key}] Bound holds: {holds}/{total} "
                  f"({100*holds/total:.0f}%)")
            print(f"    G_test mean:  {df['G_test'].mean():.4f}")
            print(f"    Bound mean:   {df[bcol].mean():.4f}")
            print(f"    Ratio median: {df[rcol].median():.1f}x")

            for model_name in df['model'].unique():
                msub = df[df['model'] == model_name]
                h = msub[hcol].sum()
                print(f"\n    {model_name}: {h}/{len(msub)}")
                for ratio in [0.3, 0.7]:
                    rsub = msub[msub['ratio'] == ratio]
                    if len(rsub) == 0:
                        continue
                    print(f"      ratio={ratio}:")
                    for method in ['k_medoids', 'k_center', 'graph_cut']:
                        s = rsub[rsub['method'] == method]
                        if len(s) == 0:
                            continue
                        print(f"        {method:12s}: "
                              f"G={s['G_test'].mean():.3f}  "
                              f"Bound={s[bcol].mean():.3f}  "
                              f"ratio={s[rcol].mean():.1f}x  "
                              f"eps={s['eps_disc'].mean():.3f}")

        # Single-model bound summary
        print(f"\n{'='*70}")
        print("SINGLE-MODEL BOUND: MAE_test(f_core) <= MAE^w_C(f_core) + L_core*[W1_tt+W1_tc]")
        print(f"{'='*70}")
        sh95 = df['single_holds_95'].sum()
        shmax = df['single_holds_max'].sum()
        print(f"  L95  holds: {sh95}/{len(df)} ({100*sh95/len(df):.0f}%)")
        print(f"  Lmax holds: {shmax}/{len(df)} ({100*shmax/len(df):.0f}%)")
        print(f"  gen_gap (MAE_test - MAE^w_C) mean: "
              f"{df['single_gen_gap'].mean():.4f}")
        print(f"  L_core*W1 (L95) mean: "
              f"{df['single_LW1_95'].mean():.4f}")
        print(f"  Coverage (L*W1 / gen_gap) mean: "
              f"{df['single_coverage_95'].mean():.2%}")
        for method in ['k_medoids', 'k_center', 'graph_cut']:
            s = df[df['method'] == method]
            sh = s['single_holds_95'].sum()
            print(f"    {method:12s}: holds={sh}/{len(s)}  "
                  f"gen_gap={s['single_gen_gap'].mean():.3f}  "
                  f"L*W1={s['single_LW1_95'].mean():.4f}  "
                  f"coverage={s['single_coverage_95'].mean():.2%}")

        # (A)(B)(C) decomposition summary
        print(f"\n{'='*70}")
        print("(A)(B)(C) DECOMPOSITION SUMMARY")
        print(f"{'='*70}")
        for model_name in df['model'].unique():
            msub = df[df['model'] == model_name]
            print(f"\n  {model_name}:")
            print(f"    (A) |MAE_test - MAE_train|(f_train) = "
                  f"{msub['term_A'].mean():.4f}  "
                  f"[bound: L_train*W1_tt = {msub['bound_A_95'].mean():.4f}]  "
                  f"holds: {(msub['term_A'] <= msub['bound_A_95'] + 1e-6).sum()}/{len(msub)}")
            for ratio in [0.3, 0.7]:
                rsub = msub[msub['ratio'] == ratio]
                if len(rsub) == 0:
                    continue
                print(f"    --- ratio={ratio} ---")
                for method in ['k_medoids', 'k_center', 'graph_cut']:
                    s = rsub[rsub['method'] == method]
                    if len(s) == 0:
                        continue
                    print(f"      {method:12s}: "
                          f"A={s['term_A'].mean():.3f}  "
                          f"B={s['term_B'].mean():.3f}  "
                          f"C={s['term_C'].mean():.3f}  "
                          f"| bA={s['bound_A_95'].mean():.4f}  "
                          f"bB={s['bound_B_95'].mean():.3f}  "
                          f"bC={s['bound_C_95'].mean():.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
