"""
Noise Behavior Analysis: Denoising Failure Patterns + Attention Under Noise.

Analysis 1: Encoder denoising failure by noise type
  - Which noise types/configs cause the most reconstruction error?
  - Per-node analysis: which nodes fail most?

Analysis 2: Gate behavior (GatedMLPEncoder)
  - Gate values g(x) on clean vs noisy inputs
  - Does the gate correctly activate for noisy nodes and stay quiet for clean?

Analysis 3: STAEformer spatial attention under noise
  - How does attention to noisy nodes change with/without encoder?
  - Does the encoder prevent attention distortion?

Usage:
  python experiments/analyze_noise_behavior.py --gpu 1 --analysis denoising_failure
  python experiments/analyze_noise_behavior.py --gpu 1 --analysis gate_behavior
  python experiments/analyze_noise_behavior.py --gpu 1 --analysis attention
  python experiments/analyze_noise_behavior.py --gpu 1 --analysis all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'noise_behavior_results')
NUM_NODES = 893
SEED = 42
DEVICE = "cuda:1"

# Noise configs to test (subset for detailed analysis)
ANALYSIS_NOISE_CONFIGS = [
    ('gaussian', 0.30, 0.3, 'gauss_s03_r30'),
    ('gaussian', 0.30, 1.0, 'gauss_s10_r30'),
    ('bias', 0.30, 0.3, 'bias_s03_r30'),
    ('bias', 0.30, 0.5, 'bias_s05_r30'),
    ('stuck', 0.30, 0, 'stuck_r30'),
    ('drift', 0.30, 0.3, 'drift_s03_r30'),
    ('drift', 0.30, 0.5, 'drift_s05_r30'),
]


# ---- Noise Injection (from eval_noise_vulnerability.py) ----
def apply_noise(inputs, corrupt_nodes, noise_type, severity, physical_channels, rng):
    """Apply noise to specific nodes. Returns corrupted tensor."""
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]

    if noise_type == 'gaussian':
        for ch in physical_channels:
            ch_std = corrupted[:, :, :, ch].std().item()
            noise_std = severity * ch_std
            noise = torch.tensor(
                rng.normal(0, noise_std, (B, T, len(corrupt_nodes))),
                dtype=corrupted.dtype)
            corrupted[:, :, corrupt_nodes, ch] += noise
        for ch in physical_channels:
            corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)

    elif noise_type == 'bias':
        for ch in physical_channels:
            for i, node in enumerate(corrupt_nodes):
                sign = 1 if rng.random() > 0.5 else -1
                factor = 1 + sign * severity
                corrupted[:, :, node, ch] *= factor

    elif noise_type == 'stuck':
        for ch in physical_channels:
            for node in corrupt_nodes:
                first_val = corrupted[:, 0, node, ch].unsqueeze(1)
                corrupted[:, :, node, ch] = first_val.expand_as(corrupted[:, :, node, ch])

    elif noise_type == 'drift':
        for ch in physical_channels:
            for i, node in enumerate(corrupt_nodes):
                sign = 1 if rng.random() > 0.5 else -1
                drift_factors = torch.linspace(1.0, 1 + sign * severity, T)
                corrupted[:, :, node, ch] *= drift_factors.unsqueeze(0)

    return corrupted


def load_data(n_batches=20):
    """Load test data (limited batches for analysis)."""
    DATA_NAME = 'SAN_BERNARDINO'
    regular_settings = get_regular_settings(DATA_NAME)

    dataset = TimeSeriesForecastingDataset(
        dataset_name=DATA_NAME,
        train_val_test_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12,
        data_range=(0, 26280),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    scaler = ZScoreScaler(
        dataset_name=DATA_NAME,
        train_ratio=regular_settings['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=regular_settings['NORM_EACH_CHANNEL'],
        rescale=regular_settings['RESCALE'],
    )

    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(batch)

    return batches, scaler


def load_functional_nodes():
    """Load functional node indices."""
    dead = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
    major_fail = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')
    functional = np.setdiff1d(np.arange(NUM_NODES), np.union1d(dead, major_fail))
    return functional, dead, major_fail


# ---- Analysis 1: Denoising Failure Patterns ----
def analyze_denoising_failure(device):
    """Analyze per-noise-type reconstruction error of the denoising encoder."""
    print("\n" + "="*80)
    print("ANALYSIS 1: Denoising Failure Patterns")
    print("="*80)

    # Load encoders
    from baselines.ContextContrastive.arch import DenoisingEncoder
    import glob

    # Load denoising v2 encoder
    encoder_v2 = DenoisingEncoder(
        input_dim=5, d_model=5, hidden_dim=64,
        temporal_layers=4, spatial_layers=1, k_neighbors=10,
        dropout=0.1, adj_path='datasets/SAN_BERNARDINO/adj_mx.pkl',
        physical_channels=[0, 1, 2], residual_connection=True,
    )
    ckpt_pattern = 'checkpoints/DenoisingPretrainV2/SAN_BERNARDINO_30_12_12/*/DenoisingPretrainV2_best_val_MAE.pt'
    ckpt = sorted(glob.glob(ckpt_pattern))[-1]
    state = torch.load(ckpt, map_location=device)
    encoder_v2.load_pretrained_weights(state.get('model_state_dict', state), strict=False)
    encoder_v2.to(device).eval()
    print(f"Loaded DenoisingEncoder v2 from {ckpt}")

    # Try loading GatedMLP encoder
    gated_mlp = None
    try:
        from baselines.ContextContrastive.arch import GatedMLPEncoder
        gated_mlp = GatedMLPEncoder(
            input_dim=5, d_model=5, hidden_dim=64, dropout=0.1,
            physical_channels=[0, 1, 2],
        )
        ckpt_pattern = 'checkpoints/GatedMLP_Pretrain/SAN_BERNARDINO_30_12_12/*/GatedMLP_Pretrain_best_val_MAE.pt'
        ckpt = sorted(glob.glob(ckpt_pattern))[-1]
        state = torch.load(ckpt, map_location=device)
        gated_mlp.load_pretrained_weights(state.get('model_state_dict', state), strict=False)
        gated_mlp.to(device).eval()
        print(f"Loaded GatedMLPEncoder from {ckpt}")
    except Exception as e:
        print(f"GatedMLP not available: {e}")

    batches, scaler = load_data(n_batches=30)
    functional, dead, major_fail = load_functional_nodes()
    physical_channels = [0, 1, 2]
    rng = np.random.RandomState(SEED)

    encoders = {'denoising_v2': encoder_v2}
    if gated_mlp is not None:
        encoders['gated_mlp'] = gated_mlp

    results = {}

    for noise_type, rate, severity, label in ANALYSIS_NOISE_CONFIGS:
        print(f"\n--- {label} ---")

        for enc_name, encoder in encoders.items():
            all_clean_error = []
            all_noisy_error = []
            all_recon_error_noisy = []  # encoder output vs clean, on noisy nodes
            all_recon_error_clean = []  # encoder output vs clean, on clean nodes

            for batch in batches:
                future, history = batch['target'], batch['inputs']
                inputs_norm = history.float()

                # Select corrupt nodes
                n_corrupt = max(1, int(rate * len(functional)))
                corrupt_nodes = rng.choice(functional, n_corrupt, replace=False)
                clean_nodes = np.setdiff1d(functional, corrupt_nodes)

                # Apply noise
                noisy_inputs = apply_noise(
                    inputs_norm, corrupt_nodes, noise_type, severity,
                    physical_channels, rng)

                # Encode
                with torch.no_grad():
                    clean_encoded = encoder.encode(inputs_norm.to(device)).cpu()
                    noisy_encoded = encoder.encode(noisy_inputs.to(device)).cpu()

                # Reconstruction error on physical channels (flow=ch0)
                # Compare encoder(noisy) vs clean original
                for ch in [0]:  # focus on flow
                    # Error on noisy nodes: how well did encoder reconstruct?
                    err_noisy = torch.abs(
                        noisy_encoded[:, :, corrupt_nodes, ch] -
                        inputs_norm[:, :, corrupt_nodes, ch]
                    ).mean().item()
                    all_recon_error_noisy.append(err_noisy)

                    # Error on clean nodes: did encoder damage clean data?
                    err_clean = torch.abs(
                        noisy_encoded[:, :, clean_nodes, ch] -
                        inputs_norm[:, :, clean_nodes, ch]
                    ).mean().item()
                    all_recon_error_clean.append(err_clean)

                    # Raw noise magnitude (before encoding)
                    noise_mag = torch.abs(
                        noisy_inputs[:, :, corrupt_nodes, ch] -
                        inputs_norm[:, :, corrupt_nodes, ch]
                    ).mean().item()
                    all_clean_error.append(noise_mag)

            avg_noise_mag = np.mean(all_clean_error)
            avg_recon_noisy = np.mean(all_recon_error_noisy)
            avg_recon_clean = np.mean(all_recon_error_clean)

            reduction_pct = (1 - avg_recon_noisy / max(avg_noise_mag, 1e-8)) * 100

            key = f"{label}_{enc_name}"
            results[key] = {
                'noise_type': noise_type,
                'label': label,
                'encoder': enc_name,
                'raw_noise_magnitude': avg_noise_mag,
                'recon_error_noisy_nodes': avg_recon_noisy,
                'recon_error_clean_nodes': avg_recon_clean,
                'noise_reduction_pct': reduction_pct,
            }

            print(f"  [{enc_name}] raw_noise={avg_noise_mag:.4f}, "
                  f"recon_noisy={avg_recon_noisy:.4f} ({reduction_pct:+.1f}% reduction), "
                  f"spillover_clean={avg_recon_clean:.4f}")

    return results


# ---- Analysis 2: Gate Behavior ----
def analyze_gate_behavior(device):
    """Analyze GatedMLPEncoder gate values on clean vs noisy data."""
    print("\n" + "="*80)
    print("ANALYSIS 2: Gate Behavior (GatedMLPEncoder)")
    print("="*80)

    from baselines.ContextContrastive.arch import GatedMLPEncoder
    import glob

    encoder = GatedMLPEncoder(
        input_dim=5, d_model=5, hidden_dim=64, dropout=0.1,
        physical_channels=[0, 1, 2],
    )
    ckpt_pattern = 'checkpoints/GatedMLP_Pretrain/SAN_BERNARDINO_30_12_12/*/GatedMLP_Pretrain_best_val_MAE.pt'
    ckpt = sorted(glob.glob(ckpt_pattern))[-1]
    state = torch.load(ckpt, map_location=device)
    encoder.load_pretrained_weights(state.get('model_state_dict', state), strict=False)
    encoder.to(device).eval()
    print(f"Loaded GatedMLPEncoder from {ckpt}")

    batches, scaler = load_data(n_batches=30)
    functional, dead, major_fail = load_functional_nodes()
    physical_channels = [0, 1, 2]
    rng = np.random.RandomState(SEED)

    results = {}

    # First: gate values on CLEAN data
    print("\n--- Clean data gate values ---")
    all_gate_clean = []
    for batch in batches:
        inputs_norm = batch['inputs'].float().to(device)
        with torch.no_grad():
            x_phys = inputs_norm[..., physical_channels]
            gate = encoder.gate_mlp(x_phys).cpu()
        # gate: [B, T, N, 3]
        all_gate_clean.append(gate[:, :, functional, :].numpy())

    gate_clean = np.concatenate(all_gate_clean, axis=0)
    print(f"  Mean gate (functional): {gate_clean.mean():.4f}")
    print(f"  Std gate (functional): {gate_clean.std():.4f}")
    print(f"  Per-channel: flow={gate_clean[..., 0].mean():.4f}, "
          f"occ={gate_clean[..., 1].mean():.4f}, "
          f"speed={gate_clean[..., 2].mean():.4f}")

    results['clean'] = {
        'gate_mean': float(gate_clean.mean()),
        'gate_std': float(gate_clean.std()),
        'gate_per_channel': [float(gate_clean[..., i].mean()) for i in range(3)],
    }

    # Gate on dead/major_fail nodes
    all_gate_dead = []
    for batch in batches:
        inputs_norm = batch['inputs'].float().to(device)
        with torch.no_grad():
            x_phys = inputs_norm[..., physical_channels]
            gate = encoder.gate_mlp(x_phys).cpu()
        if len(dead) > 0:
            all_gate_dead.append(gate[:, :, dead, :].numpy())

    if all_gate_dead:
        gate_dead = np.concatenate(all_gate_dead, axis=0)
        print(f"  Mean gate (dead nodes): {gate_dead.mean():.4f}")
        results['dead_nodes'] = {'gate_mean': float(gate_dead.mean())}

    # Now: gate values under noise
    for noise_type, rate, severity, label in ANALYSIS_NOISE_CONFIGS:
        print(f"\n--- {label} ---")

        all_gate_noisy_nodes = []
        all_gate_clean_nodes = []
        all_delta_noisy = []
        all_delta_clean = []

        for batch in batches:
            inputs_norm = batch['inputs'].float()

            n_corrupt = max(1, int(rate * len(functional)))
            corrupt_nodes = rng.choice(functional, n_corrupt, replace=False)
            clean_nodes = np.setdiff1d(functional, corrupt_nodes)

            noisy_inputs = apply_noise(
                inputs_norm, corrupt_nodes, noise_type, severity,
                physical_channels, rng)

            with torch.no_grad():
                x_phys = noisy_inputs[..., physical_channels].to(device)
                gate = encoder.gate_mlp(x_phys).cpu()
                delta = encoder.correction_mlp(x_phys).cpu()

            all_gate_noisy_nodes.append(gate[:, :, corrupt_nodes, :].numpy())
            all_gate_clean_nodes.append(gate[:, :, clean_nodes, :].numpy())
            all_delta_noisy.append(delta[:, :, corrupt_nodes, :].abs().numpy())
            all_delta_clean.append(delta[:, :, clean_nodes, :].abs().numpy())

        gate_noisy = np.concatenate(all_gate_noisy_nodes, axis=0)
        gate_clean_n = np.concatenate(all_gate_clean_nodes, axis=0)
        delta_noisy = np.concatenate(all_delta_noisy, axis=0)
        delta_clean = np.concatenate(all_delta_clean, axis=0)

        # Effective correction = gate * delta
        eff_noisy = gate_noisy * delta_noisy
        eff_clean = gate_clean_n * delta_clean

        print(f"  Gate (noisy nodes):  mean={gate_noisy.mean():.4f}, std={gate_noisy.std():.4f}")
        print(f"  Gate (clean nodes):  mean={gate_clean_n.mean():.4f}, std={gate_clean_n.std():.4f}")
        print(f"  Gate ratio (noisy/clean): {gate_noisy.mean() / max(gate_clean_n.mean(), 1e-8):.2f}x")
        print(f"  |delta| noisy={delta_noisy.mean():.4f}, clean={delta_clean.mean():.4f}")
        print(f"  Effective correction (g*|d|) noisy={eff_noisy.mean():.4f}, clean={eff_clean.mean():.4f}")

        results[label] = {
            'gate_noisy_mean': float(gate_noisy.mean()),
            'gate_clean_mean': float(gate_clean_n.mean()),
            'gate_ratio': float(gate_noisy.mean() / max(gate_clean_n.mean(), 1e-8)),
            'delta_noisy': float(delta_noisy.mean()),
            'delta_clean': float(delta_clean.mean()),
            'eff_correction_noisy': float(eff_noisy.mean()),
            'eff_correction_clean': float(eff_clean.mean()),
        }

    return results


# ---- Analysis 3: Attention Under Noise ----
def analyze_attention_under_noise(device):
    """Analyze STAEformer spatial attention with/without encoder under noise."""
    print("\n" + "="*80)
    print("ANALYSIS 3: Spatial Attention Under Noise")
    print("="*80)

    import glob
    from baselines.STAEformer.arch import STAEformer
    from baselines.STAEformer.arch.staeformer_arch import AttentionLayer

    # Hook to capture attention weights
    attention_weights = {}

    def make_attn_hook(name):
        def hook_fn(module, input, output):
            query, key, value = input
            batch_size = query.shape[0]
            head_dim = module.head_dim

            q = module.FC_Q(query)
            k = module.FC_K(key)
            q = torch.cat(torch.split(q, head_dim, dim=-1), dim=0)
            k = torch.cat(torch.split(k, head_dim, dim=-1), dim=0)
            k = k.transpose(-1, -2)
            attn = (q @ k) / head_dim**0.5
            attn = torch.softmax(attn, dim=-1)

            # Average over heads
            attn_heads = torch.split(attn, batch_size, dim=0)
            attn_avg = torch.stack(attn_heads, dim=0).mean(0)  # [B, ..., N, N]
            attention_weights[name] = attn_avg.detach().cpu()
        return hook_fn

    # Load baseline STAEformer (with noisy training)
    model_noisy = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt = sorted(glob.glob(
        'checkpoints/STAEformer_5ch_noisy/SAN_BERNARDINO_30_12_12/*/STAEformer_5ch_noisy_best_val_MAE.pt'))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model_noisy.load_state_dict(state['model_state_dict'])
    model_noisy.to(device).eval()
    print(f"Loaded STAEformer noisy from {ckpt}")

    # Load denoising v2 encoder
    from baselines.ContextContrastive.arch import DenoisingEncoder
    encoder_v2 = DenoisingEncoder(
        input_dim=5, d_model=5, hidden_dim=64,
        temporal_layers=4, spatial_layers=1, k_neighbors=10,
        dropout=0.1, adj_path='datasets/SAN_BERNARDINO/adj_mx.pkl',
        physical_channels=[0, 1, 2], residual_connection=True,
    )
    ckpt = sorted(glob.glob(
        'checkpoints/DenoisingPretrainV2/SAN_BERNARDINO_30_12_12/*/DenoisingPretrainV2_best_val_MAE.pt'))[-1]
    state = torch.load(ckpt, map_location=device)
    encoder_v2.load_pretrained_weights(state.get('model_state_dict', state), strict=False)
    encoder_v2.to(device).eval()

    # Load STAEformer with denoising v2 + noisy
    model_denoised = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt = sorted(glob.glob(
        'checkpoints/STAEformer_5ch_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/*/STAEformer_5ch_denoising_v2_noisy_best_val_MAE.pt'))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model_denoised.load_state_dict(state['model_state_dict'])
    model_denoised.to(device).eval()
    print(f"Loaded STAEformer denoising v2 noisy")

    batches, scaler = load_data(n_batches=10)
    functional, dead, major_fail = load_functional_nodes()
    physical_channels = [0, 1, 2]
    forward_features = [0, 1, 2, 3, 4]
    rng = np.random.RandomState(SEED)

    results = {}

    for noise_type, rate, severity, label in ANALYSIS_NOISE_CONFIGS:
        print(f"\n--- {label} ---")

        attn_to_noisy_baseline = []
        attn_to_noisy_denoised = []
        attn_to_clean_baseline = []
        attn_to_clean_denoised = []

        for batch in batches[:5]:  # limit for memory
            inputs_norm = batch['inputs'].float()

            n_corrupt = max(1, int(rate * len(functional)))
            corrupt_nodes = rng.choice(functional, n_corrupt, replace=False)
            clean_nodes = np.setdiff1d(functional, corrupt_nodes)

            noisy_inputs = apply_noise(
                inputs_norm, corrupt_nodes, noise_type, severity,
                physical_channels, rng)

            history = noisy_inputs[..., forward_features].to(device)

            # --- Baseline (no encoder): attention on noisy input ---
            attention_weights.clear()
            hooks = []
            # Find spatial attention layers (may be model.spatial.attn_layers_s or model.attn_layers_s)
            spatial_layers = None
            if hasattr(model_noisy, 'spatial') and hasattr(model_noisy.spatial, 'attn_layers_s'):
                spatial_layers = model_noisy.spatial.attn_layers_s
            elif hasattr(model_noisy, 'attn_layers_s'):
                spatial_layers = model_noisy.attn_layers_s
            for i, layer in enumerate(spatial_layers):
                h = layer.attn.register_forward_hook(make_attn_hook(f'spatial_{i}'))
                hooks.append(h)

            with torch.no_grad():
                model_noisy(history_data=history, future_data=None,
                           batch_seen=0, epoch=0, train=False)

            for h in hooks:
                h.remove()

            if 'spatial_0' in attention_weights:
                # attn: [B, T, N, N] - average over B and T
                attn = attention_weights['spatial_0'].mean(dim=(0, 1))  # [N, N]
                # Attention TO noisy nodes (column-wise)
                attn_to_noisy = attn[:, corrupt_nodes].mean().item()
                attn_to_clean = attn[:, clean_nodes].mean().item()
                attn_to_noisy_baseline.append(attn_to_noisy)
                attn_to_clean_baseline.append(attn_to_clean)

            # --- With encoder: attention on denoised input ---
            attention_weights.clear()
            hooks = []
            spatial_layers_d = None
            if hasattr(model_denoised, 'spatial') and hasattr(model_denoised.spatial, 'attn_layers_s'):
                spatial_layers_d = model_denoised.spatial.attn_layers_s
            elif hasattr(model_denoised, 'attn_layers_s'):
                spatial_layers_d = model_denoised.attn_layers_s
            for i, layer in enumerate(spatial_layers_d):
                h = layer.attn.register_forward_hook(make_attn_hook(f'spatial_{i}'))
                hooks.append(h)

            with torch.no_grad():
                denoised_history = encoder_v2.encode(history)
                model_denoised(history_data=denoised_history, future_data=None,
                              batch_seen=0, epoch=0, train=False)

            for h in hooks:
                h.remove()

            if 'spatial_0' in attention_weights:
                attn = attention_weights['spatial_0'].mean(dim=(0, 1))
                attn_to_noisy = attn[:, corrupt_nodes].mean().item()
                attn_to_clean = attn[:, clean_nodes].mean().item()
                attn_to_noisy_denoised.append(attn_to_noisy)
                attn_to_clean_denoised.append(attn_to_clean)

        if attn_to_noisy_baseline:
            baseline_ratio = np.mean(attn_to_noisy_baseline) / max(np.mean(attn_to_clean_baseline), 1e-8)
            denoised_ratio = np.mean(attn_to_noisy_denoised) / max(np.mean(attn_to_clean_denoised), 1e-8)

            print(f"  Baseline: attn_to_noisy={np.mean(attn_to_noisy_baseline):.6f}, "
                  f"attn_to_clean={np.mean(attn_to_clean_baseline):.6f}, "
                  f"ratio={baseline_ratio:.3f}x")
            print(f"  Denoised: attn_to_noisy={np.mean(attn_to_noisy_denoised):.6f}, "
                  f"attn_to_clean={np.mean(attn_to_clean_denoised):.6f}, "
                  f"ratio={denoised_ratio:.3f}x")

            results[label] = {
                'baseline_attn_noisy': float(np.mean(attn_to_noisy_baseline)),
                'baseline_attn_clean': float(np.mean(attn_to_clean_baseline)),
                'baseline_ratio': float(baseline_ratio),
                'denoised_attn_noisy': float(np.mean(attn_to_noisy_denoised)),
                'denoised_attn_clean': float(np.mean(attn_to_clean_denoised)),
                'denoised_ratio': float(denoised_ratio),
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['denoising_failure', 'gate_behavior', 'attention', 'all'])
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    if args.analysis in ('denoising_failure', 'all'):
        all_results['denoising_failure'] = analyze_denoising_failure(device)

    if args.analysis in ('gate_behavior', 'all'):
        all_results['gate_behavior'] = analyze_gate_behavior(device)

    if args.analysis in ('attention', 'all'):
        all_results['attention'] = analyze_attention_under_noise(device)

    # Save results
    out_path = os.path.join(OUTPUT_DIR, 'noise_behavior_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
