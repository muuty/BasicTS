"""
Validate Dual-Path Hypothesis:
1. Reliability score analysis: Does contrastive reliability detect each noise type?
2. Denoising quality comparison: DenoisingEncoder (graph conv) vs ContrastiveEncoder (temporal only)

Uses existing pretrained checkpoints — no new training needed.
"""

import os, sys, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

NUM_NODES = 893
SEED = 42


def load_contrastive_encoder(device):
    from baselines.ContextContrastive.arch import ContrastiveReliabilityEncoder
    encoder = ContrastiveReliabilityEncoder(
        input_dim=5, d_model=5, hidden_dim=32,
        temporal_layers=4, num_heads=4, dropout=0.1,
        physical_channels=[0, 1, 2], reliability_scale_init=5.0,
    )
    ckpt_path = sorted(glob.glob(
        'checkpoints/ContrastiveReliabilityPretrain/SAN_BERNARDINO_30_12_12/*/ContrastiveReliabilityPretrain_best_val_MAE.pt'))[-1]
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_pretrained_weights(state.get('model_state_dict', state), strict=False)
    encoder.to(device).eval()
    print(f"Loaded ContrastiveReliabilityEncoder from {ckpt_path}")
    return encoder


def load_denoising_encoder(device):
    from baselines.ContextContrastive.arch import DenoisingEncoder
    encoder = DenoisingEncoder(
        input_dim=5, d_model=5, hidden_dim=32,
        temporal_layers=4, spatial_layers=1, k_neighbors=10,
        dropout=0.1, adj_path='datasets/SAN_BERNARDINO/adj_mx.pkl',
        physical_channels=[0, 1, 2],
    )
    ckpt_path = sorted(glob.glob(
        'checkpoints/DenoisingPretrain/SAN_BERNARDINO_30_12_12/*/DenoisingPretrain_best_val_MAE.pt'))[-1]
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_pretrained_weights(state.get('model_state_dict', state), strict=False)
    encoder.to(device).eval()
    print(f"Loaded DenoisingEncoder from {ckpt_path}")
    return encoder


def apply_noise(inputs, corrupt_nodes, noise_type, severity, physical_channels, rng):
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]
    n = len(corrupt_nodes)
    if noise_type == 'gaussian':
        for ch in physical_channels:
            ch_std = corrupted[:, :, :, ch].std().item()
            noise = torch.tensor(rng.normal(0, severity * ch_std, (B, T, n)), dtype=corrupted.dtype)
            corrupted[:, :, corrupt_nodes, ch] += noise
            corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)
    elif noise_type == 'bias':
        factors = np.ones(n)
        under = rng.random(n) < 0.5
        factors[under] = 1.0 - severity
        factors[~under] = 1.0 + severity
        factors_t = torch.tensor(factors, dtype=corrupted.dtype).unsqueeze(0).unsqueeze(0)
        for ch in physical_channels:
            corrupted[:, :, corrupt_nodes, ch] *= factors_t
            corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)
    elif noise_type == 'stuck':
        for ch in physical_channels:
            frozen = corrupted[:, 0:1, corrupt_nodes, ch]
            corrupted[:, :, corrupt_nodes, ch] = frozen.expand_as(corrupted[:, :, corrupt_nodes, ch])
    elif noise_type == 'drift':
        directions = np.ones(n)
        directions[rng.random(n) < 0.5] = -1.0
        t_factors = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1)
        drift = torch.tensor(directions, dtype=corrupted.dtype).unsqueeze(0).unsqueeze(0)
        multiplier = 1.0 + drift * severity * t_factors
        for ch in physical_channels:
            corrupted[:, :, corrupt_nodes, ch] *= multiplier
            corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)
    return corrupted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    # Load data
    rs = get_regular_settings('SAN_BERNARDINO')
    dataset = TimeSeriesForecastingDataset(
        dataset_name='SAN_BERNARDINO',
        train_val_test_ratio=rs['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12, data_range=(0, 26280),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    scaler = ZScoreScaler(
        dataset_name='SAN_BERNARDINO',
        train_ratio=rs['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=rs['NORM_EACH_CHANNEL'],
        rescale=rs['RESCALE'],
    )
    flow_mean, flow_std = scaler.mean.float(), scaler.std.float()

    # Load functional nodes
    dead = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
    major_fail = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')
    functional = np.setdiff1d(np.arange(NUM_NODES), np.union1d(dead, major_fail))

    # Select corrupt nodes (30% of functional)
    rng_select = np.random.RandomState(SEED + 30)
    corrupt_nodes = np.sort(rng_select.choice(functional, int(len(functional) * 0.3), replace=False))
    healthy = np.setdiff1d(functional, corrupt_nodes)
    print(f"Nodes: {len(functional)} functional, {len(corrupt_nodes)} corrupt, {len(healthy)} healthy")

    # Collect raw inputs (first 500 samples for speed)
    all_inputs = []
    for batch in loader:
        all_inputs.append(batch['inputs'].float())
        if sum(x.shape[0] for x in all_inputs) >= 500:
            break
    inputs_raw = torch.cat(all_inputs, dim=0)[:500]
    print(f"Using {inputs_raw.shape[0]} samples")

    # Normalize
    inputs_norm = inputs_raw.clone()
    inputs_norm[..., 0] = (inputs_norm[..., 0] - flow_mean) / flow_std

    # Load encoders
    contrastive_enc = load_contrastive_encoder(device)
    denoising_enc = load_denoising_encoder(device)

    physical_channels = [0, 1, 2]
    noise_configs = [
        ('gaussian', 0.3, 'Gaussian s=0.3'),
        ('gaussian', 0.5, 'Gaussian s=0.5'),
        ('gaussian', 1.0, 'Gaussian s=1.0'),
        ('bias', 0.3, 'Bias s=0.3'),
        ('bias', 0.5, 'Bias s=0.5'),
        ('stuck', 0.0, 'Stuck'),
        ('drift', 0.3, 'Drift s=0.3'),
    ]

    # ========================================
    # Validation 1: Reliability scores
    # ========================================
    print("\n" + "=" * 80)
    print("VALIDATION 1: Reliability Score Analysis")
    print("=" * 80)

    # Clean reliability
    with torch.no_grad():
        _, rel_clean = contrastive_enc.encode(
            inputs_norm[..., :5].to(device), return_reliability=True)
    rel_clean = rel_clean.cpu()  # [B, T, N, 1]
    rel_clean_mean = rel_clean[:, :, :, 0].mean(dim=(0, 1))  # [N]

    print(f"\n{'Condition':<20} | {'Corrupt r':<12} | {'Healthy r':<12} | {'Gap':<10} | {'Corrupt Δ from clean':<22}")
    print("-" * 80)
    r_corrupt_clean = rel_clean_mean[corrupt_nodes].mean().item()
    r_healthy_clean = rel_clean_mean[healthy].mean().item()
    print(f"{'Clean':<20} | {r_corrupt_clean:<12.4f} | {r_healthy_clean:<12.4f} | {r_corrupt_clean - r_healthy_clean:<10.4f} | {'(baseline)':<22}")

    for noise_type, severity, label in noise_configs:
        rng_noise = np.random.RandomState(SEED + hash(label) % 10000)
        noisy = apply_noise(inputs_raw.clone(), corrupt_nodes, noise_type, severity, physical_channels, rng_noise)
        noisy_norm = noisy.clone()
        noisy_norm[..., 0] = (noisy_norm[..., 0] - flow_mean) / flow_std

        with torch.no_grad():
            _, rel_noisy = contrastive_enc.encode(
                noisy_norm[..., :5].to(device), return_reliability=True)
        rel_noisy = rel_noisy.cpu()
        rel_noisy_mean = rel_noisy[:, :, :, 0].mean(dim=(0, 1))

        r_corrupt = rel_noisy_mean[corrupt_nodes].mean().item()
        r_healthy = rel_noisy_mean[healthy].mean().item()
        delta = r_corrupt - r_corrupt_clean
        print(f"{label:<20} | {r_corrupt:<12.4f} | {r_healthy:<12.4f} | {r_corrupt - r_healthy:<10.4f} | {delta:<+22.4f}")

    # ========================================
    # Validation 2: Denoising quality
    # ========================================
    print("\n" + "=" * 80)
    print("VALIDATION 2: Denoising Quality Comparison")
    print("=" * 80)
    print(f"\n{'Condition':<20} | {'Contrastive MAE':<18} | {'Denoising MAE':<18} | {'Winner':<12}")
    print("-" * 75)

    batch_size = 50
    for noise_type, severity, label in noise_configs:
        rng_noise = np.random.RandomState(SEED + hash(label) % 10000)
        noisy = apply_noise(inputs_raw.clone(), corrupt_nodes, noise_type, severity, physical_channels, rng_noise)
        noisy_norm = noisy.clone()
        noisy_norm[..., 0] = (noisy_norm[..., 0] - flow_mean) / flow_std

        # Ground truth (clean, normalized flow channel)
        clean_flow = inputs_norm[:, :12, corrupt_nodes, 0]  # input window

        contrastive_mae_list = []
        denoising_mae_list = []

        for start in range(0, noisy_norm.shape[0], batch_size):
            end = min(start + batch_size, noisy_norm.shape[0])
            batch_noisy = noisy_norm[start:end, ..., :5].to(device)
            batch_clean_flow = clean_flow[start:end].to(device)

            with torch.no_grad():
                out_c = contrastive_enc.encode(batch_noisy)
                out_d = denoising_enc.encode(batch_noisy)

            # Compare flow channel (ch 0) on corrupt nodes
            c_mae = (out_c[:, :12, corrupt_nodes, 0] - batch_clean_flow).abs().mean().item()
            d_mae = (out_d[:, :12, corrupt_nodes, 0] - batch_clean_flow).abs().mean().item()
            contrastive_mae_list.append(c_mae)
            denoising_mae_list.append(d_mae)

        c_avg = np.mean(contrastive_mae_list)
        d_avg = np.mean(denoising_mae_list)
        winner = "Denoising" if d_avg < c_avg else "Contrastive"
        ratio = c_avg / d_avg if d_avg > 0 else float('inf')
        print(f"{label:<20} | {c_avg:<18.4f} | {d_avg:<18.4f} | {winner:<12} ({ratio:.2f}x)")

    print("\nDone.")


if __name__ == '__main__':
    main()
