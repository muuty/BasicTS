"""
Interpretability Analysis for InputSpilloverCorrector.

Analyzes:
  (a) Reliability r vs sensor health (dead/major_fail/functional)
  (b) Reliability under different noise types (corrupt vs clean separability)
  (c) Correction delta patterns
  (d) Cross-attention weights (anomaly propagation)

Usage:
  python experiments/analyze_input_corrector.py
  python experiments/analyze_input_corrector.py --ckpt checkpoints/InputCorrectorPretrain60ep/...
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from torch.utils.data import DataLoader

DEVICE = "cuda:1"
NUM_NODES = 893
DATA_NAME = 'SAN_BERNARDINO'


def load_encoder(ckpt_path, device):
    from baselines.ContextContrastive.arch import InputSpilloverCorrector
    encoder = InputSpilloverCorrector(
        input_dim=5, d_model=5, hidden_dim=64, n_heads=4,
        temporal_layers=2, dropout=0.1, physical_channels=[0, 1, 2],
        residual_connection=True,
    )
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    encoder.load_pretrained_weights(state_dict, strict=False)
    encoder.to(device).eval()
    return encoder


def load_test_data(device):
    regular_settings = get_regular_settings(DATA_NAME)
    dataset = TimeSeriesForecastingDataset(
        dataset_name=DATA_NAME,
        train_val_test_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'],
        input_len=12, output_len=12,
        mode='test', data_range=(0, 26280),
    )
    scaler = ZScoreScaler(
        dataset_name=DATA_NAME,
        train_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=regular_settings['NORM_EACH_CHANNEL'],
        rescale=regular_settings['RESCALE'],
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader, scaler


def load_sensor_categories():
    base = 'datasets/xtraffic/SAN_BERNARDINO'
    dead = np.load(os.path.join(base, 'dead_indices.npy'))
    major = np.load(os.path.join(base, 'major_fail_indices.npy'))
    functional = np.load(os.path.join(base, 'keep_no_dead_major.npy'))
    return dead, major, functional


def inject_noise(x, noise_type, rate=0.3, severity=0.5, physical_channels=[0, 1, 2]):
    """Inject noise, return (x_noisy, corrupt_mask[N])."""
    B, T, N, C = x.shape
    device = x.device
    n_corrupt = max(1, int(N * rate))
    corrupt_idx = torch.randperm(N, device=device)[:n_corrupt]
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[corrupt_idx] = True

    x_noisy = x.clone()
    if noise_type == 'gaussian':
        for ch in physical_channels:
            ch_std = x[:, :, :, ch].std().clamp(min=1e-6)
            noise = torch.randn(B, T, n_corrupt, device=device) * (severity * ch_std)
            x_noisy[:, :, corrupt_idx, ch] += noise
    elif noise_type == 'bias':
        factors = torch.ones(n_corrupt, device=device)
        under = torch.rand(n_corrupt, device=device) < 0.5
        factors[under] = 1.0 - severity
        factors[~under] = 1.0 + severity
        for ch in physical_channels:
            x_noisy[:, :, corrupt_idx, ch] *= factors.unsqueeze(0).unsqueeze(0)
    elif noise_type == 'stuck':
        for ch in physical_channels:
            frozen = x[:, 0:1, corrupt_idx, ch]
            x_noisy[:, :, corrupt_idx, ch] = frozen.expand(B, T, n_corrupt)
    elif noise_type == 'drift':
        directions = torch.ones(n_corrupt, device=device)
        directions[torch.rand(n_corrupt, device=device) < 0.5] = -1.0
        t_ramp = torch.linspace(0, 1, T, device=device).view(1, T, 1)
        multiplier = 1.0 + directions.view(1, 1, n_corrupt) * severity * t_ramp
        for ch in physical_channels:
            x_noisy[:, :, corrupt_idx, ch] *= multiplier
    elif noise_type == 'dead':
        for ch in physical_channels:
            x_noisy[:, :, corrupt_idx, ch] = 0.0

    return x_noisy, mask


def analyze_clean_reliability(encoder, loader, scaler, dead_idx, major_idx, func_idx, device):
    """(a) Reliability on clean data vs sensor health categories."""
    print("\n=== (a) Reliability vs Sensor Health (Clean Data) ===")
    all_r = []
    all_delta = []

    for batch in loader:
        history = batch['inputs'].to(device)
        history = scaler.transform(history)
        x = history[..., [0, 1, 2, 3, 4]]

        with torch.no_grad():
            _, intermediates = encoder.encode(x, return_intermediates=True)
        all_r.append(intermediates['reliability'].squeeze(-1).cpu())  # [B, N]
        all_delta.append(intermediates['delta'].cpu())  # [B, N, 3]

    r = torch.cat(all_r, dim=0).numpy()  # [total_B, N]
    delta = torch.cat(all_delta, dim=0).numpy()  # [total_B, N, 3]

    r_mean = r.mean(axis=0)  # [N]
    delta_mag = np.abs(delta).mean(axis=(0, 2))  # [N]

    cats = {
        'Dead': dead_idx,
        'Major Fail': major_idx,
        'Functional': func_idx,
    }
    for name, idx in cats.items():
        r_cat = r_mean[idx]
        d_cat = delta_mag[idx]
        print(f"  {name:12s} (n={len(idx):>3d}): r={r_cat.mean():.4f} +/- {r_cat.std():.4f}, "
              f"delta_mag={d_cat.mean():.4f} +/- {d_cat.std():.4f}")

    # Separability: functional r vs dead r
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(r_mean[func_idx], r_mean[dead_idx], alternative='greater')
    print(f"\n  Mann-Whitney U (functional r > dead r): p={p:.2e}")

    return r_mean, delta_mag


def analyze_noise_reliability(encoder, loader, scaler, device):
    """(b) Reliability under different noise types: corrupt vs clean separability."""
    print("\n=== (b) Reliability Under Noise (corrupt vs clean) ===")
    noise_types = ['gaussian', 'bias', 'stuck', 'drift', 'dead']

    # Get one batch for analysis
    batch = next(iter(loader))
    history = batch['inputs'].to(device)
    history = scaler.transform(history)
    x = history[..., [0, 1, 2, 3, 4]]

    torch.manual_seed(42)

    print(f"  {'Noise':<12} {'r_corrupt':>12} {'r_clean':>12} {'gap':>8} {'AUROC':>8}")
    print(f"  {'-'*52}")

    for nt in noise_types:
        x_noisy, mask = inject_noise(x, nt, rate=0.3, severity=0.5)
        with torch.no_grad():
            _, intermediates = encoder.encode(x_noisy, return_intermediates=True)
        r = intermediates['reliability'].squeeze(-1).cpu().numpy()  # [B, N]

        r_mean_per_node = r.mean(axis=0)  # [N]
        corrupt_nodes = mask.cpu().numpy()
        clean_nodes = ~corrupt_nodes

        r_corrupt = r_mean_per_node[corrupt_nodes].mean()
        r_clean = r_mean_per_node[clean_nodes].mean()
        gap = r_clean - r_corrupt

        # Simple AUROC
        from sklearn.metrics import roc_auc_score
        labels = clean_nodes.astype(int)
        auroc = roc_auc_score(labels, r_mean_per_node)

        print(f"  {nt:<12} {r_corrupt:>12.4f} {r_clean:>12.4f} {gap:>+8.4f} {auroc:>8.4f}")


def analyze_noise_delta(encoder, loader, scaler, device):
    """(c) Correction delta patterns under noise."""
    print("\n=== (c) Correction Delta Under Noise ===")
    noise_types = ['gaussian', 'bias', 'stuck', 'drift', 'dead']

    batch = next(iter(loader))
    history = batch['inputs'].to(device)
    history = scaler.transform(history)
    x = history[..., [0, 1, 2, 3, 4]]

    # Clean delta
    with torch.no_grad():
        _, clean_inter = encoder.encode(x, return_intermediates=True)
    clean_delta_mag = np.abs(clean_inter['delta'].cpu().numpy()).mean(axis=(0, 2))  # [N]

    torch.manual_seed(42)

    print(f"  {'Noise':<12} {'delta_corrupt':>14} {'delta_clean':>14} {'ratio':>8}")
    print(f"  {'-'*50}")
    print(f"  {'clean':<12} {clean_delta_mag.mean():>14.4f} {clean_delta_mag.mean():>14.4f} {'1.00':>8}")

    for nt in noise_types:
        x_noisy, mask = inject_noise(x, nt, rate=0.3, severity=0.5)
        with torch.no_grad():
            _, intermediates = encoder.encode(x_noisy, return_intermediates=True)
        delta = intermediates['delta'].cpu().numpy()  # [B, N, 3]
        delta_mag = np.abs(delta).mean(axis=(0, 2))  # [N]

        corrupt = mask.cpu().numpy()
        d_corrupt = delta_mag[corrupt].mean()
        d_clean = delta_mag[~corrupt].mean()
        ratio = d_corrupt / (d_clean + 1e-8)

        print(f"  {nt:<12} {d_corrupt:>14.4f} {d_clean:>14.4f} {ratio:>8.2f}")


def analyze_attention_weights(encoder, loader, scaler, device):
    """(d) Cross-attention weights: anomaly propagation patterns."""
    print("\n=== (d) Cross-Attention Weights ===")

    batch = next(iter(loader))
    history = batch['inputs'].to(device)
    history = scaler.transform(history)
    x = history[..., [0, 1, 2, 3, 4]]

    torch.manual_seed(42)

    # Dead noise (most dramatic)
    x_noisy, mask = inject_noise(x, 'dead', rate=0.1, severity=0)
    with torch.no_grad():
        _, intermediates = encoder.encode(x_noisy, return_intermediates=True)

    attn = intermediates['attn_weights']  # [B, N, N] or None
    if attn is None:
        print("  Attention weights not available")
        return

    attn = attn.cpu().numpy()  # [B, N, N]
    attn_mean = attn.mean(axis=0)  # [N, N]
    corrupt = mask.cpu().numpy()

    # How much attention do nodes pay to corrupt vs clean nodes?
    attn_to_corrupt = attn_mean[:, corrupt].sum(axis=1)  # [N] total attn each node pays to corrupt
    attn_to_clean = attn_mean[:, ~corrupt].sum(axis=1)  # [N]

    n_corrupt = corrupt.sum()
    n_clean = (~corrupt).sum()

    # Per-node average attention
    avg_attn_to_corrupt = attn_to_corrupt / n_corrupt
    avg_attn_to_clean = attn_to_clean / n_clean

    print(f"  Dead noise (rate=0.1, {n_corrupt} corrupt nodes):")
    print(f"    Avg attn per corrupt node:  {avg_attn_to_corrupt.mean():.6f}")
    print(f"    Avg attn per clean node:    {avg_attn_to_clean.mean():.6f}")
    print(f"    Ratio (corrupt/clean):      {avg_attn_to_corrupt.mean() / (avg_attn_to_clean.mean() + 1e-10):.4f}")

    # Clean nodes attention to corrupt
    clean_attn_to_corrupt = avg_attn_to_corrupt[~corrupt].mean()
    corrupt_attn_to_corrupt = avg_attn_to_corrupt[corrupt].mean()
    print(f"    Clean nodes→corrupt avg:    {clean_attn_to_corrupt:.6f}")
    print(f"    Corrupt nodes→corrupt avg:  {corrupt_attn_to_corrupt:.6f}")

    # Also check clean baseline
    with torch.no_grad():
        _, clean_inter = encoder.encode(x, return_intermediates=True)
    clean_attn = clean_inter['attn_weights']
    if clean_attn is not None:
        clean_attn = clean_attn.cpu().numpy().mean(axis=0)
        attn_entropy = -(clean_attn * np.log(clean_attn + 1e-10)).sum(axis=1)
        print(f"\n  Clean data attention entropy: {attn_entropy.mean():.4f} (max={np.log(NUM_NODES):.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Find checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        pattern = 'checkpoints/InputCorrectorPretrain/SAN_BERNARDINO_30_12_12/*/InputCorrectorPretrain_best_val_MAE.pt'
        matches = sorted(glob.glob(pattern))
        ckpt_path = matches[-1]
    print(f"Checkpoint: {ckpt_path}")

    encoder = load_encoder(ckpt_path, device)
    loader, scaler = load_test_data(device)
    dead_idx, major_idx, func_idx = load_sensor_categories()

    print(f"Sensor categories: {len(dead_idx)} dead, {len(major_idx)} major_fail, {len(func_idx)} functional")

    r_mean, delta_mag = analyze_clean_reliability(encoder, loader, scaler, dead_idx, major_idx, func_idx, device)
    analyze_noise_reliability(encoder, loader, scaler, device)
    analyze_noise_delta(encoder, loader, scaler, device)
    analyze_attention_weights(encoder, loader, scaler, device)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
