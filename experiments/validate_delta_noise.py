"""
Validate perturbation-based Δ with NOISE INJECTION on functional nodes.

Previous experiment (validate_delta.py) masked already-dead nodes → trivially no effect.
This experiment injects realistic noise into FUNCTIONAL nodes at test time,
measuring how much damage propagates to their neighbors.

Key question: When a healthy sensor suddenly breaks at test time,
how much does it hurt its neighbors' predictions?

Usage:
  python experiments/validate_delta_noise.py --gpu 1
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy import stats
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.STAEformer.arch import STAEformer
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

# Reuse noise functions from eval_noise_vulnerability.py
from experiments.eval_noise_vulnerability import (
    apply_gaussian_noise, apply_bias, apply_stuck, apply_drift, apply_dead,
)

NUM_NODES = 893
DATA_NAME = 'SAN_BERNARDINO'
PHYSICAL_CHANNELS = [0, 1, 2]


def load_model(device):
    model = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt = 'checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/50637b58eb0e35770d311d9f7bdaa214/STAEformer_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def get_topk_neighbors(adj, top_k):
    neighbors = {}
    for v in range(adj.shape[0]):
        row = adj[v].copy()
        row[v] = 0.0
        topk_idx = np.argsort(row)[-top_k:]
        topk_idx = topk_idx[row[topk_idx] > 0]
        neighbors[v] = topk_idx
    return neighbors


def load_adj():
    adj_path = f'datasets/xtraffic/{DATA_NAME}/adj_mx.pkl'
    with open(adj_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if isinstance(data, list) and len(data) >= 3:
        return np.array(data[2], dtype=np.float32)
    return np.array(data, dtype=np.float32)


def load_categories():
    base = f'datasets/xtraffic/{DATA_NAME}'
    cats = {}
    dead = np.load(os.path.join(base, 'dead_indices.npy')).astype(int)
    major = np.load(os.path.join(base, 'major_fail_indices.npy')).astype(int)
    cats['dead'] = dead
    cats['major_fail'] = major
    keep_path = os.path.join(base, 'keep_no_dead_major.npy')
    if os.path.exists(keep_path):
        cats['functional'] = np.load(keep_path).astype(int)
    return cats


def build_dataloader(n_samples):
    rs = get_regular_settings(DATA_NAME)
    dataset = TimeSeriesForecastingDataset(
        DATA_NAME, rs['TRAIN_VAL_TEST_RATIO'], 'test',
        rs['INPUT_LEN'], rs['OUTPUT_LEN'], data_range=(0, 26280),
    )
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)
    scaler = ZScoreScaler(
        DATA_NAME, rs['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=rs['NORM_EACH_CHANNEL'], rescale=rs['RESCALE'],
    )
    return loader, scaler


@torch.no_grad()
def compute_per_node_mae(model, loader, scaler, device, noise_fn=None,
                         corrupt_nodes=None, severity=0.3, rng=None):
    """Forward pass with optional noise injection, return per-node MAE (N,).

    noise_fn: one of apply_gaussian_noise, apply_stuck, etc.
    corrupt_nodes: list of node indices to corrupt.
    """
    all_ae = []
    for batch in loader:
        inputs = batch['inputs'].float()
        target = batch['target'].float()

        # Inject noise into input if requested
        if noise_fn is not None and corrupt_nodes is not None:
            inputs = noise_fn(inputs, corrupt_nodes, severity, PHYSICAL_CHANNELS, rng)

        # Normalize flow channel
        inputs_norm = inputs.clone()
        inputs_norm[..., :1] = scaler.transform(inputs[..., :1])

        inputs_norm = inputs_norm.to(device)
        out = model(inputs_norm, None, batch_seen=0, epoch=0, train=False)
        pred_raw = scaler.inverse_transform(out['prediction']).cpu()
        target_raw = target[..., :1]

        ae = torch.abs(pred_raw - target_raw)[..., 0]  # (B, T, N)
        all_ae.append(ae.mean(dim=1))  # (B, N)

    all_ae = torch.cat(all_ae, dim=0)
    return all_ae.mean(dim=0).numpy()


def print_section(title):
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-nodes', type=int, default=20,
                        help='Functional nodes to sample per noise type')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    print(f'Device: {device}')

    model = load_model(device)
    adj = load_adj()
    neighbors = get_topk_neighbors(adj, top_k=args.top_k)
    categories = load_categories()
    loader, scaler = build_dataloader(args.n_samples)

    func_indices = categories['functional']
    print(f'Functional nodes: {len(func_indices)}')
    print(f'Top-K neighbors: {args.top_k}')

    rng = np.random.RandomState(42)
    sampled_func = rng.choice(func_indices, min(args.n_nodes, len(func_indices)), replace=False)

    # =====================================================================
    # Step 1: Clean baseline
    # =====================================================================
    print_section('1. CLEAN BASELINE')
    mae_clean = compute_per_node_mae(model, loader, scaler, device)
    print(f'Clean MAE: mean={mae_clean.mean():.3f}, median={np.median(mae_clean):.3f}')
    print(f'Clean MAE (functional only): {mae_clean[func_indices].mean():.3f}')

    # =====================================================================
    # Step 2: Per-noise-type Δ on FUNCTIONAL nodes
    # =====================================================================
    noise_configs = [
        ('dead',     apply_dead,     0.0, 'Dead (zero-out)'),
        ('stuck',    apply_stuck,    0.0, 'Stuck (frozen t=0)'),
        ('gaussian', apply_gaussian_noise, 0.5, 'Gaussian (σ=0.5)'),
        ('gaussian', apply_gaussian_noise, 1.0, 'Gaussian (σ=1.0)'),
        ('bias',     apply_bias,     0.3, 'Bias (±30%)'),
        ('drift',    apply_drift,    0.5, 'Drift (50%)'),
    ]

    print_section('2. SINGLE-NODE NOISE → NEIGHBOR Δ (functional nodes)')
    print(f'Sampling {len(sampled_func)} functional nodes, measuring top-{args.top_k} neighbor MAE change\n')

    all_results = {}

    for noise_name, noise_fn, severity, label in noise_configs:
        deltas_self = []   # MAE change of the corrupted node itself
        deltas_nbr = []    # MAE change of neighbors

        for v in tqdm(sampled_func, desc=label, leave=False):
            v = int(v)
            nbrs = neighbors.get(v, np.array([]))
            mae_noisy = compute_per_node_mae(
                model, loader, scaler, device,
                noise_fn=noise_fn, corrupt_nodes=[v], severity=severity, rng=rng,
            )
            deltas_self.append(mae_noisy[v] - mae_clean[v])
            if len(nbrs) > 0:
                deltas_nbr.append(float((mae_noisy[nbrs] - mae_clean[nbrs]).mean()))

        self_arr = np.array(deltas_self)
        nbr_arr = np.array(deltas_nbr)

        all_results[noise_name + f'_s{severity}'] = {
            'label': label,
            'self_mean': self_arr.mean(),
            'self_std': self_arr.std(),
            'nbr_mean': nbr_arr.mean(),
            'nbr_std': nbr_arr.std(),
            'nbr_min': nbr_arr.min(),
            'nbr_max': nbr_arr.max(),
            'nbr_pct_positive': (nbr_arr > 0).mean() * 100,
        }

        print(f'{label:<25} '
              f'self Δ={self_arr.mean():>+8.3f}±{self_arr.std():.3f}  '
              f'nbr Δ={nbr_arr.mean():>+8.4f}±{nbr_arr.std():.4f}  '
              f'nbr>0: {(nbr_arr > 0).mean()*100:.0f}%  '
              f'[{nbr_arr.min():>+.4f}, {nbr_arr.max():>+.4f}]')

    # =====================================================================
    # Step 3: Collective noise (corrupt many functional at once)
    # =====================================================================
    print_section('3. COLLECTIVE NOISE (30% functional nodes)')

    n_corrupt = int(len(func_indices) * 0.3)
    corrupt_batch = rng.choice(func_indices, n_corrupt, replace=False).tolist()
    remaining_func = [v for v in func_indices if v not in corrupt_batch]

    for noise_name, noise_fn, severity, label in noise_configs:
        mae_noisy = compute_per_node_mae(
            model, loader, scaler, device,
            noise_fn=noise_fn, corrupt_nodes=corrupt_batch, severity=severity, rng=rng,
        )
        delta_remaining = mae_noisy[remaining_func].mean() - mae_clean[remaining_func].mean()
        delta_corrupted = mae_noisy[corrupt_batch].mean() - mae_clean[corrupt_batch].mean()
        print(f'{label:<25} '
              f'corrupted Δ={delta_corrupted:>+8.3f}  '
              f'remaining Δ={delta_remaining:>+8.4f}')

    # =====================================================================
    # Step 4: Attention vs vulnerability correlation
    # =====================================================================
    print_section('4. ATTENTION RECEIVED vs NOISE VULNERABILITY')

    # Extract attention received per node
    spatial_attn_layer = model.spatial.attn_layers_s[0].attn
    attn_accum = torch.zeros(NUM_NODES, device=device)
    count = 0
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        inputs = batch['inputs'].float()
        inputs_norm = inputs.clone()
        inputs_norm[..., :1] = scaler.transform(inputs[..., :1])
        inputs_norm = inputs_norm.to(device)
        _ = model(inputs_norm, None, batch_seen=0, epoch=0, train=False)
        # last_attn: (num_heads*B, T, N, N) → attention received = mean over row dim
        attn_weight = spatial_attn_layer.last_attn
        attn_received = attn_weight.mean(dim=(0, 1, 2))  # (N,) — avg attention each node receives
        attn_accum += attn_received
        count += 1
    attn_received_avg = (attn_accum / count).detach().cpu().numpy()

    # Correlate with stuck-noise Δ (self)
    stuck_self = []
    for v in sampled_func:
        v = int(v)
        mae_noisy = compute_per_node_mae(
            model, loader, scaler, device,
            noise_fn=apply_stuck, corrupt_nodes=[v], severity=0.0, rng=rng,
        )
        stuck_self.append(mae_noisy[v] - mae_clean[v])

    attn_sampled = attn_received_avg[sampled_func]
    stuck_self_arr = np.array(stuck_self)

    rho_attn_vuln, pval = stats.spearmanr(attn_sampled, stuck_self_arr)
    print(f'Attention received vs self-vulnerability (stuck):')
    print(f'  Spearman ρ = {rho_attn_vuln:.4f} (p={pval:.3e})')

    # Correlate clean MAE with vulnerability
    mae_sampled = mae_clean[sampled_func]
    rho_mae_vuln, pval2 = stats.spearmanr(mae_sampled, stuck_self_arr)
    print(f'Clean MAE vs self-vulnerability (stuck):')
    print(f'  Spearman ρ = {rho_mae_vuln:.4f} (p={pval2:.3e})')

    # =====================================================================
    # Summary
    # =====================================================================
    print_section('SUMMARY')
    print('Single-node noise → neighbor Δ:')
    for key, res in all_results.items():
        print(f'  {res["label"]:<25}: nbr Δ={res["nbr_mean"]:>+.4f} (>0: {res["nbr_pct_positive"]:.0f}%)')

    print(f'\nKey question: Does corrupting a functional node hurt its neighbors?')
    max_nbr = max(res['nbr_mean'] for res in all_results.values())
    if max_nbr > 0.05:
        print(f'  YES — max nbr Δ = {max_nbr:.4f}')
        print(f'  → Δ is a viable supervision signal for uncertainty')
    elif max_nbr > 0.01:
        print(f'  WEAK — max nbr Δ = {max_nbr:.4f}')
        print(f'  → Signal exists but may need amplification (collective/event-conditioned)')
    else:
        print(f'  NO — max nbr Δ = {max_nbr:.4f}')
        print(f'  → Model is too robust; spatial dependency too distributed for per-node Δ')


if __name__ == '__main__':
    main()
