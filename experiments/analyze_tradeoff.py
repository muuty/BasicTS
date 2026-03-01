"""
Robustness-Accuracy Tradeoff Analysis for InputSpilloverCorrector.

Compares 3 pretrained encoders (30ep, 60ep, wClean) on:
  1. Proxy metrics: CDR, r-AUROC, Attention Concentration
  2. Per-noise-type tradeoff decomposition
  3. Channel-level δ analysis
  4. Visualizations: r distribution, δ magnitude, CDR charts

Usage:
  python experiments/analyze_tradeoff.py --gpu 1
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:1"
NUM_NODES = 893
DATA_NAME = 'SAN_BERNARDINO'
NOISE_TYPES = ['gaussian', 'bias', 'stuck', 'drift', 'dead']
NOISE_RATE = 0.3
NOISE_SEVERITY = 0.5
PHYSICAL_CHANNELS = [0, 1, 2]
CHANNEL_NAMES = ['flow', 'occ', 'speed']
FIG_DIR = 'experiments/figures'

# Encoder checkpoint patterns
ENCODERS = {
    '30ep': 'checkpoints/InputCorrectorPretrain/SAN_BERNARDINO_30_12_12/*/InputCorrectorPretrain_best_val_MAE.pt',
    '60ep': 'checkpoints/InputCorrectorPretrain60ep/SAN_BERNARDINO_60_12_12/*/InputCorrectorPretrain60ep_best_val_MAE.pt',
    'wClean': 'checkpoints/InputCorrectorPretrainWClean/SAN_BERNARDINO_30_12_12/*/InputCorrectorPretrainWClean_best_val_MAE.pt',
}


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


def load_test_data():
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


def inject_noise(x, noise_type, rate=NOISE_RATE, severity=NOISE_SEVERITY):
    B, T, N, C = x.shape
    device = x.device
    n_corrupt = max(1, int(N * rate))
    corrupt_idx = torch.randperm(N, device=device)[:n_corrupt]
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[corrupt_idx] = True

    x_noisy = x.clone()
    if noise_type == 'gaussian':
        for ch in PHYSICAL_CHANNELS:
            ch_std = x[:, :, :, ch].std().clamp(min=1e-6)
            noise = torch.randn(B, T, n_corrupt, device=device) * (severity * ch_std)
            x_noisy[:, :, corrupt_idx, ch] += noise
    elif noise_type == 'bias':
        factors = torch.ones(n_corrupt, device=device)
        under = torch.rand(n_corrupt, device=device) < 0.5
        factors[under] = 1.0 - severity
        factors[~under] = 1.0 + severity
        for ch in PHYSICAL_CHANNELS:
            x_noisy[:, :, corrupt_idx, ch] *= factors.unsqueeze(0).unsqueeze(0)
    elif noise_type == 'stuck':
        for ch in PHYSICAL_CHANNELS:
            frozen = x[:, 0:1, corrupt_idx, ch]
            x_noisy[:, :, corrupt_idx, ch] = frozen.expand(B, T, n_corrupt)
    elif noise_type == 'drift':
        directions = torch.ones(n_corrupt, device=device)
        directions[torch.rand(n_corrupt, device=device) < 0.5] = -1.0
        t_ramp = torch.linspace(0, 1, T, device=device).view(1, T, 1)
        multiplier = 1.0 + directions.view(1, 1, n_corrupt) * severity * t_ramp
        for ch in PHYSICAL_CHANNELS:
            x_noisy[:, :, corrupt_idx, ch] *= multiplier
    elif noise_type == 'dead':
        for ch in PHYSICAL_CHANNELS:
            x_noisy[:, :, corrupt_idx, ch] = 0.0

    return x_noisy, mask


def extract_intermediates(encoder, x, device):
    """Extract r, delta, attn_weights from encoder."""
    with torch.no_grad():
        _, inter = encoder.encode(x, return_intermediates=True)
    return {
        'r': inter['reliability'].squeeze(-1).cpu().numpy(),       # [B, N]
        'delta': inter['delta'].cpu().numpy(),                      # [B, N, 3]
        'attn': inter['attn_weights'].cpu().numpy() if inter['attn_weights'] is not None else None,
    }


def compute_metrics_for_encoder(encoder, x_batch, device):
    """Compute all proxy metrics for one encoder on one batch.

    Returns dict with:
      - clean: intermediates on clean data
      - per_noise: {noise_type: {intermediates, mask, CDR, AUROC, attn_conc}}
    """
    # Clean intermediates
    clean = extract_intermediates(encoder, x_batch, device)
    clean_delta_mag = np.abs(clean['delta']).mean(axis=(0, 2))  # [N]
    clean_delta_per_ch = np.abs(clean['delta']).mean(axis=0)     # [N, 3]

    results = {
        'clean_r_mean': clean['r'].mean(axis=0),         # [N]
        'clean_delta_mag': clean_delta_mag,               # [N]
        'clean_delta_per_ch': clean_delta_per_ch,         # [N, 3]
        'clean_r_global': clean['r'].mean(),
        'clean_delta_global': clean_delta_mag.mean(),
        'noise': {},
    }

    torch.manual_seed(42)

    for nt in NOISE_TYPES:
        x_noisy, mask = inject_noise(x_batch, nt)
        noisy = extract_intermediates(encoder, x_noisy, device)

        corrupt = mask.cpu().numpy()
        noisy_delta_mag = np.abs(noisy['delta']).mean(axis=(0, 2))  # [N]
        noisy_delta_per_ch = np.abs(noisy['delta']).mean(axis=0)     # [N, 3]
        r_per_node = noisy['r'].mean(axis=0)  # [N]

        # CDR: correction discrimination ratio
        d_corrupt = noisy_delta_mag[corrupt].mean()
        d_clean = noisy_delta_mag[~corrupt].mean()
        cdr = d_corrupt / (d_clean + 1e-8)

        # r-AUROC
        labels = (~corrupt).astype(int)  # clean=1, corrupt=0
        auroc = roc_auc_score(labels, r_per_node)

        # Attention concentration
        attn_conc = None
        if noisy['attn'] is not None:
            attn_mean = noisy['attn'].mean(axis=0)  # [N, N]
            n_c = corrupt.sum()
            n_h = (~corrupt).sum()
            avg_attn_corrupt = attn_mean[:, corrupt].sum(axis=1) / n_c
            avg_attn_clean = attn_mean[:, ~corrupt].sum(axis=1) / n_h
            attn_conc = avg_attn_corrupt.mean() / (avg_attn_clean.mean() + 1e-10)

        # r stats
        r_corrupt = r_per_node[corrupt].mean()
        r_clean = r_per_node[~corrupt].mean()

        results['noise'][nt] = {
            'cdr': cdr,
            'auroc': auroc,
            'attn_conc': attn_conc,
            'r_corrupt': r_corrupt,
            'r_clean': r_clean,
            'delta_corrupt': d_corrupt,
            'delta_clean_nodes': d_clean,
            'delta_per_ch_corrupt': noisy_delta_per_ch[corrupt].mean(axis=0),  # [3]
            'delta_per_ch_clean': noisy_delta_per_ch[~corrupt].mean(axis=0),   # [3]
        }

    return results


def print_comparison_table(all_results):
    """Print formatted comparison of proxy metrics across encoders."""
    names = list(all_results.keys())

    print("\n" + "=" * 90)
    print("PROXY METRICS COMPARISON")
    print("=" * 90)

    # Clean data
    print(f"\n{'Metric':<25}", end="")
    for name in names:
        print(f"{name:>20}", end="")
    print()
    print("-" * (25 + 20 * len(names)))

    print(f"{'Clean r (global mean)':<25}", end="")
    for name in names:
        print(f"{all_results[name]['clean_r_global']:>20.4f}", end="")
    print()

    print(f"{'Clean |δ| (global mean)':<25}", end="")
    for name in names:
        print(f"{all_results[name]['clean_delta_global']:>20.4f}", end="")
    print()

    # Channel-level clean delta
    for ci, ch_name in enumerate(CHANNEL_NAMES):
        print(f"{'  δ_clean (' + ch_name + ')':<25}", end="")
        for name in names:
            val = all_results[name]['clean_delta_per_ch'].mean(axis=0)[ci]
            print(f"{val:>20.4f}", end="")
        print()

    # Per-noise metrics
    for nt in NOISE_TYPES:
        print(f"\n--- {nt.upper()} (rate={NOISE_RATE}, sev={NOISE_SEVERITY}) ---")
        header_line = f"{'Metric':<25}"
        for name in names:
            header_line += f"{name:>20}"
        print(header_line)
        print("-" * (25 + 20 * len(names)))

        for metric, label in [('cdr', 'CDR'), ('auroc', 'r-AUROC'),
                               ('attn_conc', 'Attn Concentration'),
                               ('r_corrupt', 'r (corrupt nodes)'),
                               ('r_clean', 'r (clean nodes)'),
                               ('delta_corrupt', '|δ| corrupt'),
                               ('delta_clean_nodes', '|δ| clean nodes')]:
            print(f"{label:<25}", end="")
            for name in names:
                val = all_results[name]['noise'][nt][metric]
                if val is None:
                    print(f"{'N/A':>20}", end="")
                else:
                    print(f"{val:>20.4f}", end="")
            print()

        # Channel-level delta for corrupt nodes
        for ci, ch_name in enumerate(CHANNEL_NAMES):
            print(f"{'  δ_corrupt (' + ch_name + ')':<25}", end="")
            for name in names:
                val = all_results[name]['noise'][nt]['delta_per_ch_corrupt'][ci]
                print(f"{val:>20.4f}", end="")
            print()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY: AVG ACROSS NOISE TYPES")
    print("=" * 90)
    print(f"{'Metric':<25}", end="")
    for name in names:
        print(f"{name:>20}", end="")
    print()
    print("-" * (25 + 20 * len(names)))

    for metric, label in [('cdr', 'Avg CDR'), ('auroc', 'Avg r-AUROC'),
                           ('attn_conc', 'Avg Attn Conc')]:
        print(f"{label:<25}", end="")
        for name in names:
            vals = [all_results[name]['noise'][nt][metric]
                    for nt in NOISE_TYPES
                    if all_results[name]['noise'][nt][metric] is not None]
            print(f"{np.mean(vals):>20.4f}" if vals else f"{'N/A':>20}", end="")
        print()

    # Clean pass-through
    print(f"{'Clean |δ|':<25}", end="")
    for name in names:
        print(f"{all_results[name]['clean_delta_global']:>20.4f}", end="")
    print()


def plot_r_distribution(all_results, save_dir):
    """Side-by-side r distribution histograms for clean data."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4), sharey=True)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        r = res['clean_r_mean']
        ax.hist(r, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{name}\nmean={r.mean():.4f}, std={r.std():.4f}')
        ax.set_xlabel('Reliability r')
        ax.axvline(r.mean(), color='red', linestyle='--', alpha=0.7)

    axes[0].set_ylabel('Node count')
    fig.suptitle('Reliability Distribution on Clean Data', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tradeoff_r_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_dir}/tradeoff_r_distribution.png")


def plot_delta_by_noise(all_results, save_dir):
    """Bar chart of δ magnitude by noise type for each encoder."""
    names = list(all_results.keys())
    x = np.arange(len(NOISE_TYPES))
    width = 0.8 / len(names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Corrupt node delta
    for i, name in enumerate(names):
        vals = [all_results[name]['noise'][nt]['delta_corrupt'] for nt in NOISE_TYPES]
        ax1.bar(x + i * width, vals, width, label=name, alpha=0.8)
    ax1.set_xticks(x + width * (len(names) - 1) / 2)
    ax1.set_xticklabels(NOISE_TYPES)
    ax1.set_ylabel('|δ| magnitude')
    ax1.set_title('Correction Magnitude on Corrupt Nodes')
    ax1.legend()

    # CDR
    for i, name in enumerate(names):
        vals = [all_results[name]['noise'][nt]['cdr'] for nt in NOISE_TYPES]
        ax2.bar(x + i * width, vals, width, label=name, alpha=0.8)
    ax2.set_xticks(x + width * (len(names) - 1) / 2)
    ax2.set_xticklabels(NOISE_TYPES)
    ax2.set_ylabel('CDR (δ_corrupt / δ_clean)')
    ax2.set_title('Correction Discrimination Ratio')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tradeoff_delta_cdr.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_dir}/tradeoff_delta_cdr.png")


def plot_channel_delta(all_results, save_dir):
    """Channel-level δ analysis: flow vs occ vs speed."""
    names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        # Clean delta per channel
        clean_ch = res['clean_delta_per_ch'].mean(axis=0)  # [3]

        # Noisy delta per channel (average across noise types, corrupt nodes)
        noisy_ch = np.mean([res['noise'][nt]['delta_per_ch_corrupt'] for nt in NOISE_TYPES], axis=0)

        x = np.arange(3)
        ax.bar(x - 0.2, clean_ch, 0.35, label='Clean', alpha=0.7, color='steelblue')
        ax.bar(x + 0.2, noisy_ch, 0.35, label='Noisy (avg)', alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_title(f'{name}')
        ax.legend()

    axes[0].set_ylabel('|δ| magnitude')
    fig.suptitle('Channel-Level Correction: Clean vs Noisy', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tradeoff_channel_delta.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_dir}/tradeoff_channel_delta.png")


def plot_auroc_comparison(all_results, save_dir):
    """AUROC comparison by noise type."""
    names = list(all_results.keys())
    x = np.arange(len(NOISE_TYPES))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        vals = [all_results[name]['noise'][nt]['auroc'] for nt in NOISE_TYPES]
        ax.bar(x + i * width, vals, width, label=name, alpha=0.8)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(NOISE_TYPES)
    ax.set_ylabel('AUROC')
    ax.set_title('Noise Detection: r-AUROC by Noise Type')
    ax.legend()
    ax.set_ylim(0.3, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tradeoff_auroc.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_dir}/tradeoff_auroc.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load test data
    print("Loading test data...")
    loader, scaler = load_test_data()

    # Get a representative batch (use multiple for robustness)
    batches = []
    for i, batch in enumerate(loader):
        if i >= 5:  # 5 batches = 320 samples
            break
        batches.append(batch['inputs'].to(device))
    x_all = torch.cat(batches, dim=0)
    x_all = scaler.transform(x_all)
    x_all = x_all[..., [0, 1, 2, 3, 4]]
    print(f"  Using {x_all.shape[0]} test samples, shape {x_all.shape}")

    # Load and analyze each encoder
    all_results = {}
    for name, pattern in ENCODERS.items():
        matches = sorted(glob.glob(pattern))
        ckpt_path = matches[-1]
        print(f"\n{'='*60}")
        print(f"Encoder: {name}")
        print(f"  Checkpoint: {ckpt_path}")

        encoder = load_encoder(ckpt_path, device)
        results = compute_metrics_for_encoder(encoder, x_all, device)
        all_results[name] = results

        del encoder
        torch.cuda.empty_cache()

    # Print comparison
    print_comparison_table(all_results)

    # Generate visualizations
    print("\n\nGenerating figures...")
    plot_r_distribution(all_results, FIG_DIR)
    plot_delta_by_noise(all_results, FIG_DIR)
    plot_channel_delta(all_results, FIG_DIR)
    plot_auroc_comparison(all_results, FIG_DIR)

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
