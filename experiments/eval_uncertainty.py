"""
Uncertainty Quality Evaluation for STAEformerUncertainty.

Evaluates learned uncertainty (sigma) against actual prediction errors:
1. sigma-error correlation (Spearman): does sigma predict where errors are high?
2. sigma by sensor category: dead > major_fail > partial > functional?
3. Calibration: does mu +/- k*sigma cover the expected fraction of true values?
4. sigma variance: did sigma collapse to a constant?
5. (Optional) noise injection: does sigma increase for corrupted nodes?

Usage:
  python experiments/eval_uncertainty.py
  python experiments/eval_uncertainty.py --ckpt-dir checkpoints/STAEformerUncertainty_5ch/SAN_BERNARDINO_30_12_12/<hash>
  python experiments/eval_uncertainty.py --gpu 1
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.STAEformer.arch import STAEformerUncertainty
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

NUM_NODES = 893
DATA_NAME = 'SAN_BERNARDINO'


def find_best_ckpt(ckpt_dir=None):
    """Find the best checkpoint path automatically."""
    if ckpt_dir and os.path.isdir(ckpt_dir):
        best = os.path.join(ckpt_dir, 'STAEformerUncertainty_5ch_best_val_MAE.pt')
        if os.path.exists(best):
            return best
        # Try glob
        candidates = glob.glob(os.path.join(ckpt_dir, '*best_val*.pt'))
        if candidates:
            return candidates[0]

    # Auto-discover
    base = 'checkpoints/STAEformerUncertainty_5ch/SAN_BERNARDINO_30_12_12'
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            full = os.path.join(base, d)
            if os.path.isdir(full):
                candidates = glob.glob(os.path.join(full, '*best_val*.pt'))
                if candidates:
                    return candidates[0]
    return None


def load_model(ckpt_path, device):
    model = STAEformerUncertainty(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True, sigma_min=1e-3,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_sensor_categories():
    """Load sensor category indices."""
    base = f'datasets/xtraffic/{DATA_NAME}'
    cats = {}
    for name in ['dead_indices', 'major_fail_indices']:
        path = os.path.join(base, f'{name}.npy')
        if os.path.exists(path):
            cats[name.replace('_indices', '')] = np.load(path)
    # Derive partial_fail and functional
    all_nodes = set(range(NUM_NODES))
    dead = set(cats.get('dead', []))
    major = set(cats.get('major_fail', []))
    # partial fail = keep_no_dead_major complement minus dead/major
    keep_path = os.path.join(base, 'keep_no_dead_major.npy')
    if os.path.exists(keep_path):
        keep = set(np.load(keep_path))
        cats['partial_fail'] = np.array(sorted(all_nodes - dead - major - keep))
        cats['functional'] = np.array(sorted(keep))
    return cats


def build_dataloader(device):
    regular_settings = get_regular_settings(DATA_NAME)
    dataset = TimeSeriesForecastingDataset(
        dataset_name=DATA_NAME,
        train_val_test_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'],
        mode='test',
        input_len=regular_settings['INPUT_LEN'],
        output_len=regular_settings['OUTPUT_LEN'],
        data_range=(0, 26280),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    scaler = ZScoreScaler(
        dataset_name=DATA_NAME,
        train_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=regular_settings['NORM_EACH_CHANNEL'],
        rescale=regular_settings['RESCALE'],
    )
    return loader, scaler


@torch.no_grad()
def collect_predictions_simple(model, loader, scaler, device):
    """Run model on test set, collect mu, sigma, target, and absolute error.

    Works in z-score space (sufficient for ranking/correlation metrics).
    DataLoader returns dict with 'inputs' (B,T,N,C) and 'target' (B,T,N,C).
    """
    all_mu, all_sigma, all_target, all_ae = [], [], [], []

    for batch in tqdm(loader, desc='Inference'):
        inputs = batch['inputs'].float().to(device)   # (B, 12, N, 5)
        target = batch['target'].float().to(device)    # (B, 12, N, 5)

        # Normalize physical channels (0=flow, 1=occ, 2=speed)
        inputs_norm = inputs.clone()
        for ch in range(3):
            inputs_norm[..., ch] = scaler.transform(inputs[..., ch])

        # Target flow in z-score space
        target_z = scaler.transform(target[..., 0]).unsqueeze(-1)  # (B, T, N, 1)

        out = model(inputs_norm, target, batch_seen=0, epoch=0, train=False)
        mu = out['prediction']   # (B, T, N, 1) z-score
        sigma = out['sigma']     # (B, T, N, 1) z-score space

        ae = torch.abs(mu - target_z)

        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_target.append(target_z.cpu())
        all_ae.append(ae.cpu())

    return (torch.cat(all_mu), torch.cat(all_sigma),
            torch.cat(all_target), torch.cat(all_ae))


def eval_sigma_error_correlation(sigma, ae):
    """Per-node: mean sigma vs mean absolute error."""
    # sigma, ae: (N_samples, T, N_nodes, 1)
    node_sigma = sigma[..., 0].mean(dim=(0, 1)).numpy()  # (N,)
    node_ae = ae[..., 0].mean(dim=(0, 1)).numpy()        # (N,)

    rho, pval = stats.spearmanr(node_sigma, node_ae)
    return {
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
        'node_sigma_mean': float(node_sigma.mean()),
        'node_sigma_std': float(node_sigma.std()),
        'node_ae_mean': float(node_ae.mean()),
    }


def eval_sigma_by_category(sigma, ae, categories):
    """Mean sigma and MAE per sensor category."""
    node_sigma = sigma[..., 0].mean(dim=(0, 1)).numpy()
    node_ae = ae[..., 0].mean(dim=(0, 1)).numpy()
    results = {}
    for cat_name, indices in categories.items():
        if len(indices) == 0:
            continue
        results[cat_name] = {
            'n_nodes': int(len(indices)),
            'mean_sigma': float(node_sigma[indices].mean()),
            'std_sigma': float(node_sigma[indices].std()),
            'mean_ae': float(node_ae[indices].mean()),
        }
    return results


def eval_calibration(mu, sigma, target):
    """Empirical coverage at k-sigma levels."""
    residual = torch.abs(mu - target)  # (S, T, N, 1)
    results = {}
    for k, expected in [(1.0, 0.6827), (1.96, 0.95), (2.576, 0.99)]:
        within = (residual <= k * sigma).float()
        coverage = within.mean().item()
        results[f'{k:.2f}sigma'] = {
            'coverage': float(coverage),
            'expected': float(expected),
            'gap': float(coverage - expected),
        }
    return results


def eval_sigma_collapse(sigma):
    """Check if sigma collapsed to a constant."""
    node_sigma = sigma[..., 0].mean(dim=(0, 1)).numpy()
    cv = float(node_sigma.std() / (node_sigma.mean() + 1e-8))
    return {
        'sigma_mean': float(node_sigma.mean()),
        'sigma_std': float(node_sigma.std()),
        'sigma_cv': cv,
        'collapsed': cv < 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Find checkpoint
    ckpt_path = find_best_ckpt(args.ckpt_dir)
    if ckpt_path is None:
        print('ERROR: No checkpoint found. Specify --ckpt-dir or wait for training to complete.')
        return
    print(f'Checkpoint: {ckpt_path}')

    # Load model
    model = load_model(ckpt_path, device)
    print(f'Model loaded ({sum(p.numel() for p in model.parameters())} params)')

    # Load data
    loader, scaler = build_dataloader(device)
    categories = load_sensor_categories()
    print(f'Test samples: {len(loader.dataset)}, Categories: {list(categories.keys())}')

    # Collect predictions
    mu, sigma, target, ae = collect_predictions_simple(model, loader, scaler, device)
    print(f'Collected: mu={mu.shape}, sigma={sigma.shape}')

    # Evaluate
    results = {}

    print('\n=== 1. Sigma-Error Correlation ===')
    corr = eval_sigma_error_correlation(sigma, ae)
    results['sigma_error_correlation'] = corr
    print(f"  Spearman rho: {corr['spearman_rho']:.4f} (p={corr['spearman_pval']:.2e})")
    print(f"  Node sigma: mean={corr['node_sigma_mean']:.4f}, std={corr['node_sigma_std']:.4f}")

    print('\n=== 2. Sigma by Sensor Category ===')
    cat_results = eval_sigma_by_category(sigma, ae, categories)
    results['sigma_by_category'] = cat_results
    for cat, v in cat_results.items():
        print(f"  {cat:15s}: sigma={v['mean_sigma']:.4f} +/- {v['std_sigma']:.4f}, "
              f"MAE={v['mean_ae']:.4f}, n={v['n_nodes']}")

    print('\n=== 3. Calibration ===')
    cal = eval_calibration(mu, sigma, target)
    results['calibration'] = cal
    for level, v in cal.items():
        print(f"  {level}: coverage={v['coverage']:.4f} (expected={v['expected']:.4f}, "
              f"gap={v['gap']:+.4f})")

    print('\n=== 4. Sigma Collapse Check ===')
    collapse = eval_sigma_collapse(sigma)
    results['sigma_collapse'] = collapse
    print(f"  CV(sigma): {collapse['sigma_cv']:.4f} "
          f"({'COLLAPSED' if collapse['collapsed'] else 'OK'})")

    # Save
    out_dir = os.path.dirname(ckpt_path)
    out_path = os.path.join(out_dir, 'uncertainty_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out_path}')

    # Summary verdict
    print('\n=== VERDICT ===')
    ok = True
    if collapse['collapsed']:
        print('  FAIL: sigma collapsed to constant')
        ok = False
    if corr['spearman_rho'] < 0.1:
        print(f"  WARN: low sigma-error correlation (rho={corr['spearman_rho']:.3f})")
    elif corr['spearman_rho'] < 0.3:
        print(f"  MARGINAL: moderate correlation (rho={corr['spearman_rho']:.3f})")
    else:
        print(f"  GOOD: strong correlation (rho={corr['spearman_rho']:.3f})")

    # Check ordering
    cat_order = ['functional', 'partial_fail', 'major_fail', 'dead']
    sigmas = [cat_results.get(c, {}).get('mean_sigma', 0) for c in cat_order if c in cat_results]
    if sigmas == sorted(sigmas):
        print('  GOOD: sigma ordering matches sensor health (functional < ... < dead)')
    else:
        ordering = [f"{c}={cat_results[c]['mean_sigma']:.3f}" for c in cat_order if c in cat_results]
        print(f'  WARN: sigma ordering unexpected: {ordering}')

    if ok:
        print('  => Proceed to Exp B (curriculum) and Exp C (noisy training)')
    else:
        print('  => Check contingency plans in the experiment plan')


if __name__ == '__main__':
    main()
