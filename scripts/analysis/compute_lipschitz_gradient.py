#!/usr/bin/env python
"""
Gradient-based Lipschitz constant estimation for coreset forecasting models.

For each model f, computes:
    L̂_f = max_{x_i ∈ D} ||∇_{x_i} φ_f(x_i)||₂

where φ_f(x) = per-sample MAE = (1/d) Σ |f(x) - y(x)|.

Outputs per-model L statistics (L_max, L_95, L_mean) and connects
to the single-model bound:
    MAE_test(f_coreset) ≤ MAE^w(f_coreset) + L_coreset · [W₁_tt + W₁_tc]

Usage (on GPU node):
    sgpu-cuda
    conda activate cuda && python scripts/analysis/compute_lipschitz_gradient.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    'DCRNN': {
        'cfg': 'baselines/DCRNN/SAN_BERNARDINO/SAN_BERNARDINO.py',
        'full_ckpt': 'checkpoints/phase_b_full_data/DCRNN/xtraffic/'
                     'SAN_BERNARDINO_100_12_12/1/10d1d855012ec5c21d28417af93a25bb/'
                     'DCRNN_best_val_MAE.pt',
    },
    'STAEformer': {
        'cfg': 'baselines/STAEformer/SAN_BERNARDINO/SAN_BERNARDINO.py',
        'full_ckpt': 'checkpoints/phase_b_full_data/STAEformer/xtraffic/'
                     'SAN_BERNARDINO_100_12_12/1/cc2b4eb34df2f50fc87c83c1142db0cb/'
                     'STAEformer_best_val_MAE.pt',
    },
}

CKPT_BASE = Path('checkpoints/phase_b_deterministic')


# ── Model helpers ────────────────────────────────────────────────────────────

def build_model_and_scaler(cfg_path: str, device: torch.device):
    cfg = import_config(cfg_path, verbose=False)
    model = cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM']).to(device)
    model.eval()
    scaler = cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
    return model, scaler, cfg


def load_weights(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # strict=False: DCRNN checkpoints have extra cached gconv keys
    model.load_state_dict(sd, strict=False)
    model.eval()


# ── Differentiable forward pass ──────────────────────────────────────────────

def differentiable_forward_mae(model, scaler, cfg, x, target, device):
    """Forward pass with autograd support through the scaler.

    Returns per-sample MAE as a (B,) tensor with gradients attached to x.
    """
    fwd_feat = cfg['MODEL'].get('FORWARD_FEATURES', None)
    tgt_feat = cfg['MODEL'].get('TARGET_FEATURES', None)

    # ── Differentiable z-score transform on channel 0 ──
    mean = scaler.mean.to(device).float()
    std = scaler.std.to(device).float()
    # Replace channel 0 with scaled version (no in-place ops)
    ch0_scaled = (x[..., 0:1] - mean) / std
    if x.shape[-1] > 1:
        x_scaled = torch.cat([ch0_scaled, x[..., 1:]], dim=-1)
    else:
        x_scaled = ch0_scaled

    # Select forward features
    history = x_scaled[..., fwd_feat] if fwd_feat is not None else x_scaled

    # Model forward
    pred = model(history_data=history, future_data=None,
                 batch_seen=None, epoch=None, train=False)
    if isinstance(pred, dict):
        pred = pred['prediction']

    # Differentiable inverse transform on prediction
    if scaler.rescale:
        pred = pred * std + mean

    # Target
    tgt = target[..., tgt_feat] if tgt_feat is not None else target

    # Per-sample MAE
    abs_err = torch.abs(pred - tgt)
    null_val = cfg['METRICS'].get('NULL_VAL', np.nan)
    if np.isnan(null_val):
        mask = ~torch.isnan(tgt)
    else:
        mask = (tgt != null_val)
    dims = tuple(range(1, abs_err.ndim))
    counts = mask.float().sum(dim=dims).clamp(min=1)
    per_sample_mae = (abs_err * mask.float()).sum(dim=dims) / counts

    return per_sample_mae  # (B,)


# ── Gradient-based Lipschitz estimation ──────────────────────────────────────

def compute_gradient_norms(model, scaler, cfg, dataloader, device):
    """Compute ||∇_x φ_f(x_i)||₂ for all samples via backprop."""
    model.eval()
    all_grad_norms = []
    all_maes = []

    for batch in tqdm(dataloader, desc="    gradient L", leave=False):
        inputs = batch['inputs'].to(device).float()
        target = batch['target'].to(device).float()
        B = inputs.shape[0]

        # Process one sample at a time for clean per-sample gradients
        for i in range(B):
            x = inputs[i:i+1].clone().detach().requires_grad_(True)
            tgt = target[i:i+1]

            mae = differentiable_forward_mae(model, scaler, cfg, x, tgt, device)
            mae_val = mae.item()

            # Backward
            mae.backward()

            if x.grad is not None:
                grad_norm = x.grad.flatten().norm(2).item()
            else:
                grad_norm = 0.0

            all_grad_norms.append(grad_norm)
            all_maes.append(mae_val)

    return np.array(all_grad_norms), np.array(all_maes)


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
        with open(coreset_file) as f:
            info['coreset_indices'] = json.load(f)
        results.append(info)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--models', nargs='+',
                        default=['STGCNChebGraphConv', 'AGCRN', 'DCRNN', 'STAEformer'])
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    outdir = Path('experiments/result/analysis')
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Full-data models (L per architecture) ────────────────────
    print("\n" + "=" * 70)
    print("PART 1: Gradient Lipschitz for full-data models")
    print("=" * 70)

    full_model_results = []

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"WARNING: No config for {model_name}, skipping")
            continue

        mcfg = MODEL_CONFIGS[model_name]
        if not Path(mcfg['full_ckpt']).exists():
            print(f"WARNING: No checkpoint for {model_name}, skipping")
            continue

        print(f"\n--- {model_name} (full-data) ---")
        model, scaler, cfg = build_model_and_scaler(mcfg['cfg'], device)
        load_weights(model, mcfg['full_ckpt'], device)

        # Train set
        ds_params = cfg['DATASET']['PARAM']
        train_ds = TimeSeriesForecastingDataset(mode='train', **ds_params)
        test_ds = TimeSeriesForecastingDataset(mode='test', **ds_params)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        print(f"  Computing gradient norms on train set ({len(train_ds)} samples)...")
        grad_norms_train, maes_train = compute_gradient_norms(
            model, scaler, cfg, train_loader, device)

        print(f"  Computing gradient norms on test set ({len(test_ds)} samples)...")
        grad_norms_test, maes_test = compute_gradient_norms(
            model, scaler, cfg, test_loader, device)

        L_max = float(np.max(grad_norms_train))
        L_95 = float(np.percentile(grad_norms_train, 95))
        L_mean = float(np.mean(grad_norms_train))
        L_std = float(np.std(grad_norms_train))

        print(f"  L_max  = {L_max:.6f}")
        print(f"  L_95   = {L_95:.6f}")
        print(f"  L_mean = {L_mean:.6f}")
        print(f"  L_std  = {L_std:.6f}")
        print(f"  MAE_train = {np.mean(maes_train):.4f}")
        print(f"  MAE_test  = {np.mean(maes_test):.4f}")

        full_model_results.append({
            'model': model_name,
            'type': 'full_data',
            'L_max': round(L_max, 6),
            'L_95': round(L_95, 6),
            'L_mean': round(L_mean, 6),
            'L_std': round(L_std, 6),
            'mae_train': round(float(np.mean(maes_train)), 4),
            'mae_test': round(float(np.mean(maes_test)), 4),
            'n_samples': len(train_ds),
        })

        # Save per-sample gradient norms for further analysis
        np.savez(outdir / f'grad_norms_{model_name}_full.npz',
                 grad_norms_train=grad_norms_train,
                 grad_norms_test=grad_norms_test,
                 maes_train=maes_train,
                 maes_test=maes_test)

    # ── Part 2: Coreset models (L per method/ratio) ──────────────────────
    print("\n" + "=" * 70)
    print("PART 2: Gradient Lipschitz for coreset models")
    print("=" * 70)

    coreset_results = []

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            continue
        mcfg = MODEL_CONFIGS[model_name]

        ckpts = discover_coreset_checkpoints(model_name)
        if not ckpts:
            print(f"\n  {model_name}: no coreset checkpoints found")
            continue

        print(f"\n--- {model_name}: {len(ckpts)} coreset checkpoints ---")
        model, scaler, cfg = build_model_and_scaler(mcfg['cfg'], device)

        ds_params = cfg['DATASET']['PARAM']
        train_ds = TimeSeriesForecastingDataset(mode='train', **ds_params)
        test_ds = TimeSeriesForecastingDataset(mode='test', **ds_params)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        for ck in tqdm(ckpts, desc=f"  {model_name} coreset"):
            load_weights(model, ck['ckpt_path'], device)

            grad_norms_train, maes_train = compute_gradient_norms(
                model, scaler, cfg, train_loader, device)
            grad_norms_test, maes_test = compute_gradient_norms(
                model, scaler, cfg, test_loader, device)

            L_max = float(np.max(grad_norms_train))
            L_95 = float(np.percentile(grad_norms_train, 95))
            L_mean = float(np.mean(grad_norms_train))

            coreset_results.append({
                'model': model_name,
                'type': 'coreset',
                'method': ck['method'],
                'ratio': ck['ratio'],
                'distance': ck['distance'],
                'seed': ck['seed'],
                'L_max': round(L_max, 6),
                'L_95': round(L_95, 6),
                'L_mean': round(L_mean, 6),
                'mae_train': round(float(np.mean(maes_train)), 4),
                'mae_test': round(float(np.mean(maes_test)), 4),
            })

    # ── Save results ─────────────────────────────────────────────────────
    df_full = pd.DataFrame(full_model_results)
    df_core = pd.DataFrame(coreset_results)

    df_full.to_csv(outdir / 'lipschitz_gradient_full.csv', index=False)
    print(f"\nSaved full-model results: {outdir / 'lipschitz_gradient_full.csv'}")

    if len(df_core) > 0:
        df_core.to_csv(outdir / 'lipschitz_gradient_coreset.csv', index=False)
        print(f"Saved coreset results: {outdir / 'lipschitz_gradient_coreset.csv'}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: Gradient Lipschitz by Model Architecture")
    print(f"{'=' * 70}")
    for _, row in df_full.iterrows():
        print(f"  {row['model']:25s}  L_max={row['L_max']:.6f}  "
              f"L_95={row['L_95']:.6f}  L_mean={row['L_mean']:.6f}  "
              f"MAE_test={row['mae_test']:.4f}")

    if len(df_core) > 0:
        print(f"\n{'=' * 70}")
        print("SUMMARY: Coreset L by method (averaged over seeds)")
        print(f"{'=' * 70}")
        for model_name in df_core['model'].unique():
            msub = df_core[df_core['model'] == model_name]
            # Find full-data L for comparison
            full_L = df_full[df_full['model'] == model_name]['L_max'].values
            full_L = full_L[0] if len(full_L) > 0 else float('nan')
            print(f"\n  {model_name} (full L_max={full_L:.6f}):")
            for method in ['k_medoids', 'k_center', 'graph_cut']:
                s = msub[msub['method'] == method]
                if len(s) == 0:
                    continue
                print(f"    {method:12s}: L_max={s['L_max'].mean():.6f}  "
                      f"L_95={s['L_95'].mean():.6f}  "
                      f"MAE_test={s['mae_test'].mean():.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
