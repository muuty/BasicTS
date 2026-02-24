"""
Debug why contrastive reliability scores don't discriminate noisy nodes.

Checks:
1. Learned scale/bias parameters → sigmoid regime
2. Raw cosine similarities before sigmoid
3. h_node vs h_context representation analysis
4. Cross-attention pattern analysis
5. Per-timestep vs time-pooled comparison
"""

import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

NUM_NODES = 893
SEED = 42


def apply_noise(inputs, corrupt_nodes, noise_type, severity, rng):
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]
    n = len(corrupt_nodes)
    physical = [0, 1, 2]
    if noise_type == 'gaussian':
        for ch in physical:
            ch_std = corrupted[:, :, :, ch].std().item()
            noise = torch.tensor(rng.normal(0, severity * ch_std, (B, T, n)), dtype=corrupted.dtype)
            corrupted[:, :, corrupt_nodes, ch] += noise
            corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)
    elif noise_type == 'bias':
        factors = np.ones(n)
        factors[rng.random(n) < 0.5] = 1.0 - severity
        factors[~(factors < 1)] = 1.0 + severity
        ft = torch.tensor(factors, dtype=corrupted.dtype).unsqueeze(0).unsqueeze(0)
        for ch in physical:
            corrupted[:, :, corrupt_nodes, ch] *= ft
            corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)
    elif noise_type == 'stuck':
        for ch in physical:
            frozen = corrupted[:, 0:1, corrupt_nodes, ch]
            corrupted[:, :, corrupt_nodes, ch] = frozen.expand_as(corrupted[:, :, corrupt_nodes, ch])
    return corrupted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    # Load encoder
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

    # =============================================
    # Check 1: Learned parameters
    # =============================================
    print("=" * 80)
    print("CHECK 1: Learned Scale/Bias Parameters")
    print("=" * 80)
    scale = encoder.reliability_scale.item()
    bias = encoder.reliability_bias.item()
    print(f"  reliability_scale = {scale:.4f} (init was 5.0)")
    print(f"  reliability_bias  = {bias:.4f} (init was 0.0)")
    print(f"  sigmoid(scale * cos + bias) where cos ∈ [-1, 1]")
    print(f"  → sigmoid range: [{torch.sigmoid(torch.tensor(scale * (-1) + bias)).item():.4f}, "
          f"{torch.sigmoid(torch.tensor(scale * 1 + bias)).item():.4f}]")
    print(f"  → At cos=0: sigmoid({bias:.4f}) = {torch.sigmoid(torch.tensor(bias)).item():.4f}")
    print(f"  → At cos=0.5: sigmoid({scale*0.5 + bias:.4f}) = {torch.sigmoid(torch.tensor(scale*0.5 + bias)).item():.4f}")

    # Load data
    rs = get_regular_settings('SAN_BERNARDINO')
    dataset = TimeSeriesForecastingDataset(
        dataset_name='SAN_BERNARDINO',
        train_val_test_ratio=rs['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12, data_range=(0, 26280),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    scaler = ZScoreScaler(
        dataset_name='SAN_BERNARDINO',
        train_ratio=rs['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=rs['NORM_EACH_CHANNEL'], rescale=rs['RESCALE'],
    )
    flow_mean, flow_std = scaler.mean.float(), scaler.std.float()

    dead = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
    major_fail = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')
    functional = np.setdiff1d(np.arange(NUM_NODES), np.union1d(dead, major_fail))
    rng_select = np.random.RandomState(SEED + 30)
    corrupt_nodes = np.sort(rng_select.choice(functional, int(len(functional) * 0.3), replace=False))
    healthy = np.setdiff1d(functional, corrupt_nodes)

    # Get a batch
    batch = next(iter(loader))
    inputs_raw = batch['inputs'].float()[:32]
    inputs_norm = inputs_raw.clone()
    inputs_norm[..., 0] = (inputs_norm[..., 0] - flow_mean) / flow_std
    x = inputs_norm[..., :5].to(device)

    # =============================================
    # Check 2: Raw cosine similarities (clean)
    # =============================================
    print("\n" + "=" * 80)
    print("CHECK 2: Raw Cosine Similarities (Clean Input)")
    print("=" * 80)

    with torch.no_grad():
        B, T, N, D = x.shape
        x_phys = x[..., encoder.physical_channels]
        h = encoder.input_proj(x_phys)
        h = h.permute(0, 2, 3, 1).reshape(B * N, encoder.hidden_dim, T)
        for block in encoder.temporal_blocks:
            h = block(h)
        h = h.reshape(B, N, encoder.hidden_dim, T).permute(0, 3, 1, 2)
        residual = h
        h = encoder.channel_mixer(h)
        h = encoder.channel_norm(residual + h)

        # Time-pooled
        h_node = h.mean(dim=1)  # [B, N, hidden]
        h_context = encoder.cross_pred_attn(h_node)
        h_context = encoder.cross_pred_norm(h_context)

        cos_sim = F.cosine_similarity(h_node, h_context, dim=-1)  # [B, N]

    cos_mean = cos_sim.mean(dim=0).cpu()  # [N]
    print(f"  Cosine sim stats (all nodes): mean={cos_mean.mean():.4f}, std={cos_mean.std():.4f}, "
          f"min={cos_mean.min():.4f}, max={cos_mean.max():.4f}")
    print(f"  Cosine sim (functional):      mean={cos_mean[functional].mean():.4f}, std={cos_mean[functional].std():.4f}")
    print(f"  Cosine sim (dead):            mean={cos_mean[dead].mean():.4f}, std={cos_mean[dead].std():.4f}")
    print(f"  Cosine sim (corrupt subset):  mean={cos_mean[corrupt_nodes].mean():.4f}")
    print(f"  Cosine sim (healthy subset):  mean={cos_mean[healthy].mean():.4f}")

    # =============================================
    # Check 3: h_node representation analysis
    # =============================================
    print("\n" + "=" * 80)
    print("CHECK 3: Representation Analysis")
    print("=" * 80)

    h_node_cpu = h_node.mean(dim=0).cpu()  # [N, hidden]
    h_ctx_cpu = h_context.mean(dim=0).cpu()

    print(f"  h_node norms:   mean={h_node_cpu.norm(dim=-1).mean():.4f}, std={h_node_cpu.norm(dim=-1).std():.4f}")
    print(f"  h_context norms: mean={h_ctx_cpu.norm(dim=-1).mean():.4f}, std={h_ctx_cpu.norm(dim=-1).std():.4f}")

    # Inter-node similarity (are all nodes too similar?)
    h_normed = F.normalize(h_node_cpu, dim=-1)
    sim_matrix = h_normed @ h_normed.T  # [N, N]
    off_diag = sim_matrix[~torch.eye(N, dtype=torch.bool)]
    print(f"  Inter-node cosine sim: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    print(f"  → If mean ≈ 1: all nodes have very similar representations (collapse)")

    # =============================================
    # Check 4: Cross-attention patterns
    # =============================================
    print("\n" + "=" * 80)
    print("CHECK 4: Cross-Attention Patterns")
    print("=" * 80)

    with torch.no_grad():
        attn_module = encoder.cross_pred_attn
        H = attn_module.num_heads
        d = attn_module.head_dim
        Q = attn_module.W_Q(h_node).view(B, N, H, d).transpose(1, 2)
        K = attn_module.W_K(h_node).view(B, N, H, d).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / (d ** 0.5)
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        scores = scores.masked_fill(self_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)

    attn_mean = attn.mean(dim=(0, 1)).cpu()  # [N, N] averaged over batch and heads
    # Check entropy (uniform = high entropy, focused = low entropy)
    attn_entropy = -(attn_mean * (attn_mean + 1e-10).log()).sum(dim=-1)
    max_entropy = np.log(N - 1)  # uniform over N-1 nodes
    print(f"  Attention entropy: mean={attn_entropy.mean():.4f}, max_possible={max_entropy:.4f}")
    print(f"  Entropy ratio (1.0 = perfectly uniform): {(attn_entropy.mean() / max_entropy):.4f}")

    # Attention to dead vs functional
    attn_to_dead = attn_mean[:, dead].sum(dim=-1).mean().item()
    attn_to_func = attn_mean[:, functional].sum(dim=-1).mean().item()
    print(f"  Avg attention TO dead nodes: {attn_to_dead:.4f} (proportion of dead: {len(dead)/N:.4f})")
    print(f"  Avg attention TO functional: {attn_to_func:.4f} (proportion of func: {len(functional)/N:.4f})")

    # =============================================
    # Check 5: Cosine sim under noise (raw, before sigmoid)
    # =============================================
    print("\n" + "=" * 80)
    print("CHECK 5: Cosine Similarity Under Noise (Before Sigmoid)")
    print("=" * 80)

    noise_configs = [
        ('gaussian', 0.5, 'Gaussian s=0.5'),
        ('gaussian', 1.0, 'Gaussian s=1.0'),
        ('bias', 0.5, 'Bias s=0.5'),
        ('stuck', 0.0, 'Stuck'),
    ]

    print(f"\n{'Condition':<20} | {'Corrupt cos':<14} | {'Healthy cos':<14} | {'Gap':<10} | {'After sigmoid':<14}")
    print("-" * 80)

    # Clean baseline
    cos_corrupt_clean = cos_mean[corrupt_nodes].mean().item()
    cos_healthy_clean = cos_mean[healthy].mean().item()
    r_corrupt = torch.sigmoid(torch.tensor(scale * cos_corrupt_clean + bias)).item()
    r_healthy = torch.sigmoid(torch.tensor(scale * cos_healthy_clean + bias)).item()
    print(f"{'Clean':<20} | {cos_corrupt_clean:<14.6f} | {cos_healthy_clean:<14.6f} | "
          f"{cos_corrupt_clean - cos_healthy_clean:<10.6f} | {r_corrupt:.4f} vs {r_healthy:.4f}")

    for noise_type, severity, label in noise_configs:
        rng_noise = np.random.RandomState(SEED + hash(label) % 10000)
        noisy = apply_noise(inputs_raw.clone(), corrupt_nodes, noise_type, severity, rng_noise)
        noisy_norm = noisy.clone()
        noisy_norm[..., 0] = (noisy_norm[..., 0] - flow_mean) / flow_std
        x_noisy = noisy_norm[..., :5].to(device)

        with torch.no_grad():
            x_phys_n = x_noisy[..., encoder.physical_channels]
            h_n = encoder.input_proj(x_phys_n)
            h_n = h_n.permute(0, 2, 3, 1).reshape(B * N, encoder.hidden_dim, T)
            for block in encoder.temporal_blocks:
                h_n = block(h_n)
            h_n = h_n.reshape(B, N, encoder.hidden_dim, T).permute(0, 3, 1, 2)
            res_n = h_n
            h_n = encoder.channel_mixer(h_n)
            h_n = encoder.channel_norm(res_n + h_n)
            h_node_n = h_n.mean(dim=1)
            h_ctx_n = encoder.cross_pred_attn(h_node_n)
            h_ctx_n = encoder.cross_pred_norm(h_ctx_n)
            cos_n = F.cosine_similarity(h_node_n, h_ctx_n, dim=-1)

        cos_n_mean = cos_n.mean(dim=0).cpu()
        c_corrupt = cos_n_mean[corrupt_nodes].mean().item()
        c_healthy = cos_n_mean[healthy].mean().item()
        r_c = torch.sigmoid(torch.tensor(scale * c_corrupt + bias)).item()
        r_h = torch.sigmoid(torch.tensor(scale * c_healthy + bias)).item()
        delta_cos = c_corrupt - cos_corrupt_clean
        print(f"{label:<20} | {c_corrupt:<14.6f} | {c_healthy:<14.6f} | "
              f"{delta_cos:<+10.6f} | {r_c:.4f} vs {r_h:.4f}")

    # =============================================
    # Check 6: Per-timestep analysis (before pooling)
    # =============================================
    print("\n" + "=" * 80)
    print("CHECK 6: Per-Timestep vs Time-Pooled (Stuck Noise)")
    print("=" * 80)

    rng_stuck = np.random.RandomState(SEED + 999)
    stuck_input = apply_noise(inputs_raw.clone(), corrupt_nodes, 'stuck', 0, rng_stuck)
    stuck_norm = stuck_input.clone()
    stuck_norm[..., 0] = (stuck_norm[..., 0] - flow_mean) / flow_std

    with torch.no_grad():
        # Clean
        x_c = inputs_norm[..., :5].to(device)
        xp_c = x_c[..., encoder.physical_channels]
        hc = encoder.input_proj(xp_c)
        hc = hc.permute(0, 2, 3, 1).reshape(B * N, encoder.hidden_dim, T)
        for block in encoder.temporal_blocks:
            hc = block(hc)
        hc = hc.reshape(B, N, encoder.hidden_dim, T).permute(0, 3, 1, 2)  # [B, T, N, H]
        resc = hc
        hc = encoder.channel_mixer(hc)
        hc = encoder.channel_norm(resc + hc)

        # Stuck
        x_s = stuck_norm[..., :5].to(device)
        xp_s = x_s[..., encoder.physical_channels]
        hs = encoder.input_proj(xp_s)
        hs = hs.permute(0, 2, 3, 1).reshape(B * N, encoder.hidden_dim, T)
        for block in encoder.temporal_blocks:
            hs = block(hs)
        hs = hs.reshape(B, N, encoder.hidden_dim, T).permute(0, 3, 1, 2)
        ress = hs
        hs = encoder.channel_mixer(hs)
        hs = encoder.channel_norm(ress + hs)

    # Per-timestep temporal variance (how much does representation change over time?)
    # [B, T, N, H] → compute variance across T for each node
    clean_temporal_var = hc.var(dim=1).mean(dim=(0, -1)).cpu()  # [N]
    stuck_temporal_var = hs.var(dim=1).mean(dim=(0, -1)).cpu()  # [N]

    print(f"  Temporal variance of h (clean):")
    print(f"    Corrupt nodes: {clean_temporal_var[corrupt_nodes].mean():.6f}")
    print(f"    Healthy nodes: {clean_temporal_var[healthy].mean():.6f}")
    print(f"  Temporal variance of h (stuck noise on corrupt):")
    print(f"    Corrupt nodes: {stuck_temporal_var[corrupt_nodes].mean():.6f} "
          f"(ratio: {stuck_temporal_var[corrupt_nodes].mean() / clean_temporal_var[corrupt_nodes].mean():.4f}x)")
    print(f"    Healthy nodes: {stuck_temporal_var[healthy].mean():.6f}")

    # Time-pooled h_node similarity between clean and stuck
    h_node_clean = hc.mean(dim=1)  # [B, N, H]
    h_node_stuck = hs.mean(dim=1)

    cos_clean_vs_stuck = F.cosine_similarity(
        h_node_clean[:, corrupt_nodes, :],
        h_node_stuck[:, corrupt_nodes, :], dim=-1).mean().item()
    print(f"\n  Cosine(h_clean, h_stuck) for corrupt nodes: {cos_clean_vs_stuck:.6f}")
    print(f"  → If ≈ 1: time-pooled representations are nearly identical despite stuck noise")
    print(f"  → This would explain why reliability can't detect stuck")

    print("\nDone.")


if __name__ == '__main__':
    main()
