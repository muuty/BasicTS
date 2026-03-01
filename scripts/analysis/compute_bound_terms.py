#!/usr/bin/env python
"""
Compute the 3 decomposition terms of the generalization bound for each coreset setting.

(A) test-train shift:       E_test[l(f_S)] - E_train[l(f_S)]
(B) train-coreset mismatch: E_train[l(f_S)] - E_core[l(f_S)]
(C) generalization gap:     E_core[l(f_S)] - E_hat_S[l(f_S)]  (≈ 0 for subset selection)

Also estimates empirical Lipschitz constant L of f_S.

Usage:
    srun --jobid=<JOB_ID> --pty bash -c \
        "conda activate cuda && python scripts/analysis/compute_bound_terms.py --gpus 0"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easytorch.config import import_config
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.metrics import masked_mae


# ── Config mapping ──────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'STGCNChebGraphConv': 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py',
    'AGCRN': 'baselines/AGCRN/SAN_BERNARDINO/SAN_BERNARDINO.py',
}

CKPT_BASE = Path('checkpoints/phase_b_deterministic')


def parse_checkpoint_config(ckpt_dir: Path) -> dict:
    """Parse coreset setting from cfg.txt in checkpoint directory."""
    cfg_file = ckpt_dir / 'cfg.txt'
    if not cfg_file.exists():
        return None

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
            elif line.startswith('MD5:'):
                info['md5'] = line.split(':')[1].strip()
    return info


def discover_checkpoints(model_name: str) -> list:
    """Find all checkpoint directories for a model."""
    base = CKPT_BASE / model_name / 'xtraffic' / 'SAN_BERNARDINO_100_12_12' / '1'
    if not base.exists():
        return []

    results = []
    for hash_dir in sorted(base.iterdir()):
        if not hash_dir.is_dir():
            continue
        ckpt_file = hash_dir / f'{model_name}_best_val_MAE.pt'
        if not ckpt_file.exists():
            continue
        info = parse_checkpoint_config(hash_dir)
        if info is None:
            continue
        info['ckpt_path'] = str(ckpt_file)
        info['ckpt_dir'] = str(hash_dir)
        info['model'] = model_name
        results.append(info)
    return results


# ── Evaluation helpers ──────────────────────────────────────────────────────

def build_model_and_scaler(cfg_path: str, device: torch.device):
    """Build model and scaler from config, return (model, scaler, cfg)."""
    cfg = import_config(cfg_path, verbose=False)

    # Build model
    model = cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM'])
    model = model.to(device)
    model.eval()

    # Build scaler
    scaler = cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])

    return model, scaler, cfg


def load_checkpoint(model, ckpt_path: str, device: torch.device):
    """Load model weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()


def build_dataset(cfg, mode: str) -> TimeSeriesForecastingDataset:
    """Build dataset for given mode (train/test)."""
    return cfg['DATASET']['TYPE'](mode=mode, **cfg['DATASET']['PARAM'])


@torch.no_grad()
def compute_mae_on_split(model, scaler, cfg, dataloader, device) -> float:
    """Compute MAE on a given dataloader using the model's forward pass."""
    forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
    target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
    null_val = cfg['METRICS'].get('NULL_VAL', np.nan)

    all_preds = []
    all_targets = []

    for batch in dataloader:
        inputs = batch['inputs'].to(device).float()
        target = batch['target'].to(device).float()

        # Apply scaler transform
        if scaler is not None:
            inputs = scaler.transform(inputs)
            target_scaled = scaler.transform(target.clone())

        # Select forward features
        if forward_features is not None:
            history = inputs[..., forward_features]
        else:
            history = inputs

        # Model forward
        pred = model(history_data=history, future_data=None,
                     batch_seen=None, epoch=None, train=False)
        if isinstance(pred, dict):
            pred = pred['prediction']

        # Inverse transform predictions
        if scaler is not None and scaler.rescale:
            pred = scaler.inverse_transform(pred)

        # Select target features from original target
        if target_features is not None:
            target_sel = target[..., target_features]
        else:
            target_sel = target

        all_preds.append(pred.cpu())
        all_targets.append(target_sel.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mae = masked_mae(all_preds, all_targets, null_val).item()
    return mae


@torch.no_grad()
def estimate_lipschitz(model, scaler, cfg, dataset, device,
                       n_pairs: int = 2000, batch_size: int = 64) -> dict:
    """Estimate empirical Lipschitz constant by sampling random pairs.

    L = max |f(x_i) - f(x_j)| / ||x_i - x_j||
    """
    forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
    n = len(dataset)

    # Sample random pairs
    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    # Ensure different samples
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % n

    # Compute predictions for all unique indices
    all_idx = np.unique(np.concatenate([idx_a, idx_b]))
    subset = Subset(dataset, all_idx.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    preds_map = {}  # index -> prediction
    inputs_map = {}  # index -> raw input (for distance)
    offset = 0
    for batch in loader:
        inputs = batch['inputs'].to(device).float()
        if scaler is not None:
            inputs_s = scaler.transform(inputs.clone())
        else:
            inputs_s = inputs

        if forward_features is not None:
            history = inputs_s[..., forward_features]
        else:
            history = inputs_s

        pred = model(history_data=history, future_data=None,
                     batch_seen=None, epoch=None, train=False)
        if isinstance(pred, dict):
            pred = pred['prediction']

        if scaler is not None and scaler.rescale:
            pred = scaler.inverse_transform(pred)

        bs = pred.shape[0]
        for i in range(bs):
            real_idx = all_idx[offset + i]
            preds_map[real_idx] = pred[i].cpu().flatten()
            inputs_map[real_idx] = inputs[i].cpu().flatten()  # original scale
        offset += bs

    # Compute Lipschitz ratios
    ratios = []
    for a, b in zip(idx_a, idx_b):
        pred_diff = torch.norm(preds_map[a] - preds_map[b]).item()
        input_diff = torch.norm(inputs_map[a] - inputs_map[b]).item()
        if input_diff > 1e-8:
            ratios.append(pred_diff / input_diff)

    ratios = np.array(ratios)
    return {
        'L_max': float(np.max(ratios)),
        'L_99': float(np.percentile(ratios, 99)),
        'L_95': float(np.percentile(ratios, 95)),
        'L_median': float(np.median(ratios)),
        'L_mean': float(np.mean(ratios)),
        'n_pairs': len(ratios),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--models', nargs='+',
                        default=['STGCNChebGraphConv', 'AGCRN'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lipschitz-pairs', type=int, default=2000)
    parser.add_argument('--skip-lipschitz', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    outdir = Path('experiments/result/analysis')
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"WARNING: No config for {model_name}, skipping")
            continue

        cfg_path = MODEL_CONFIGS[model_name]
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"Config: {cfg_path}")
        print(f"{'='*80}")

        # Build model, scaler, config once per model type
        model, scaler, cfg = build_model_and_scaler(cfg_path, device)

        # Build datasets once (shared across all checkpoints of same model)
        print("Building datasets...")
        train_dataset = build_dataset(cfg, 'train')
        test_dataset = build_dataset(cfg, 'test')
        print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        # Discover checkpoints
        checkpoints = discover_checkpoints(model_name)
        print(f"Found {len(checkpoints)} checkpoints")

        # Filter to smart methods only
        smart_methods = {'k_medoids', 'k_center', 'graph_cut'}
        checkpoints = [c for c in checkpoints if c['method'] in smart_methods]
        print(f"Smart methods: {len(checkpoints)} checkpoints")

        # Estimate Lipschitz once per model (using first checkpoint)
        lipschitz_info = None
        if not args.skip_lipschitz and checkpoints:
            print("\nEstimating Lipschitz constant (using first checkpoint)...")
            load_checkpoint(model, checkpoints[0]['ckpt_path'], device)
            lipschitz_info = estimate_lipschitz(
                model, scaler, cfg, train_dataset, device,
                n_pairs=args.lipschitz_pairs, batch_size=args.batch_size
            )
            print(f"  L_max={lipschitz_info['L_max']:.4f}, "
                  f"L_95={lipschitz_info['L_95']:.4f}, "
                  f"L_median={lipschitz_info['L_median']:.4f}")

        for i, ckpt_info in enumerate(checkpoints):
            method = ckpt_info['method']
            distance = ckpt_info['distance']
            ratio = ckpt_info['ratio']
            seed = ckpt_info['seed']
            ckpt_path = ckpt_info['ckpt_path']
            ckpt_dir = ckpt_info['ckpt_dir']

            print(f"\n[{i+1}/{len(checkpoints)}] {method}, {distance}, "
                  f"ratio={ratio}, seed={seed}")

            # Load checkpoint
            load_checkpoint(model, ckpt_path, device)

            # Load coreset indices
            coreset_file = Path(ckpt_dir) / 'coreset-selection.json'
            if not coreset_file.exists():
                print(f"  WARNING: No coreset-selection.json, skipping")
                continue
            with open(coreset_file) as f:
                coreset_indices = json.load(f)

            # Build coreset dataloader
            coreset_subset = Subset(train_dataset, coreset_indices)
            coreset_loader = DataLoader(coreset_subset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=0)

            # Compute MAE on each split
            print(f"  Computing MAE on 3 splits...", end=" ", flush=True)

            mae_test = compute_mae_on_split(model, scaler, cfg, test_loader, device)
            mae_train = compute_mae_on_split(model, scaler, cfg, train_loader, device)
            mae_coreset = compute_mae_on_split(model, scaler, cfg, coreset_loader, device)

            # Decomposition terms
            term_A = mae_test - mae_train      # test-train shift
            term_B = mae_train - mae_coreset   # train-coreset mismatch
            # term_C ≈ 0 for subset selection
            gap = mae_test - mae_coreset        # total gap = A + B + C

            print(f"test={mae_test:.4f}, train={mae_train:.4f}, "
                  f"coreset={mae_coreset:.4f}")
            print(f"  (A)={term_A:+.4f}, (B)={term_B:+.4f}, gap={gap:+.4f}")

            result = {
                'model': model_name,
                'method': method,
                'distance': distance,
                'ratio': ratio,
                'seed': seed,
                'mae_test': round(mae_test, 4),
                'mae_train': round(mae_train, 4),
                'mae_coreset': round(mae_coreset, 4),
                'term_A_test_train': round(term_A, 4),
                'term_B_train_coreset': round(term_B, 4),
                'total_gap': round(gap, 4),
                'n_coreset': len(coreset_indices),
                'n_train': len(train_dataset),
            }
            if lipschitz_info:
                result.update({f'lipschitz_{k}': v
                               for k, v in lipschitz_info.items()})

            all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)
    out_path = outdir / 'bound_decomposition.csv'
    df.to_csv(out_path, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(df)} results to {out_path}")

    # Summary statistics
    if len(df) > 0:
        print(f"\n{'='*80}")
        print("SUMMARY: Mean term magnitudes")
        print(f"{'='*80}")
        for model_name in df['model'].unique():
            mdf = df[df['model'] == model_name]
            print(f"\n  {model_name} ({len(mdf)} settings):")
            print(f"    MAE_test:  {mdf['mae_test'].mean():.4f} ± {mdf['mae_test'].std():.4f}")
            print(f"    MAE_train: {mdf['mae_train'].mean():.4f} ± {mdf['mae_train'].std():.4f}")
            print(f"    MAE_core:  {mdf['mae_coreset'].mean():.4f} ± {mdf['mae_coreset'].std():.4f}")
            print(f"    (A) test-train:      {mdf['term_A_test_train'].mean():+.4f} ± {mdf['term_A_test_train'].std():.4f}")
            print(f"    (B) train-coreset:   {mdf['term_B_train_coreset'].mean():+.4f} ± {mdf['term_B_train_coreset'].std():.4f}")

            # Per ratio
            for ratio in [0.3, 0.7]:
                rdf = mdf[mdf['ratio'] == ratio]
                if len(rdf) > 0:
                    print(f"    --- ratio={ratio} ---")
                    print(f"      (A): {rdf['term_A_test_train'].mean():+.4f} ± {rdf['term_A_test_train'].std():.4f}")
                    print(f"      (B): {rdf['term_B_train_coreset'].mean():+.4f} ± {rdf['term_B_train_coreset'].std():.4f}")

                    # Per method
                    for method in ['k_medoids', 'k_center', 'graph_cut']:
                        sub = rdf[rdf['method'] == method]
                        if len(sub) > 0:
                            print(f"        {method:12s}: (A)={sub['term_A_test_train'].mean():+.4f}  "
                                  f"(B)={sub['term_B_train_coreset'].mean():+.4f}  "
                                  f"test={sub['mae_test'].mean():.2f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
