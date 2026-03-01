"""
Deep analysis of perturbation-based Δ — multi-angle view.

Analyses:
1. Individual node Δ distribution (not just category means)
2. Top-K sensitivity: how K changes the signal
3. Attention-based influence: spatial attention weights → who the model actually listens to
4. Multi-node masking: mask all dead nodes at once → collective effect
5. Input variance vs Δ: does input variability predict usefulness?
6. Per-sample Δ variance: is the signal consistent or noisy?

Usage:
  python experiments/validate_delta_deep.py --gpu 1
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy import stats
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.STAEformer.arch import STAEformer
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

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


def load_adj():
    adj_path = f'datasets/xtraffic/{DATA_NAME}/adj_mx.pkl'
    with open(adj_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if isinstance(data, list) and len(data) >= 3:
        adj = data[2]
    else:
        adj = data
    return np.array(adj, dtype=np.float32)


def get_topk_neighbors(adj, top_k):
    neighbors = {}
    for v in range(adj.shape[0]):
        row = adj[v].copy()
        row[v] = 0.0
        topk_idx = np.argsort(row)[-top_k:]
        topk_idx = topk_idx[row[topk_idx] > 0]
        neighbors[v] = topk_idx
    return neighbors


def load_categories():
    base = f'datasets/xtraffic/{DATA_NAME}'
    cats = {}
    dead = np.load(os.path.join(base, 'dead_indices.npy')).astype(int)
    major = np.load(os.path.join(base, 'major_fail_indices.npy')).astype(int)
    cats['dead'] = dead
    cats['major_fail'] = major
    keep_path = os.path.join(base, 'keep_no_dead_major.npy')
    if os.path.exists(keep_path):
        keep = set(np.load(keep_path).astype(int))
        all_nodes = set(range(NUM_NODES))
        partial = np.array(sorted(all_nodes - set(dead) - set(major) - keep))
        if len(partial) > 0:
            cats['partial_fail'] = partial
        cats['functional'] = np.array(sorted(keep))
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
def compute_per_node_mae(model, loader, scaler, device, mask_nodes=None):
    """Forward pass, return per-node MAE (N,) in original space.
    mask_nodes: list/array of node indices to zero out, or None.
    """
    all_ae = []
    for batch in loader:
        inputs = batch['inputs'].float()
        target = batch['target'].float()

        if mask_nodes is not None:
            inputs = inputs.clone()
            for v in mask_nodes:
                for ch in PHYSICAL_CHANNELS:
                    inputs[:, :, v, ch] = 0.0

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


@torch.no_grad()
def compute_per_node_mae_per_sample(model, loader, scaler, device, mask_node=None):
    """Return per-sample per-node MAE: (S, N)."""
    all_ae = []
    for batch in loader:
        inputs = batch['inputs'].float()
        target = batch['target'].float()

        if mask_node is not None:
            inputs = inputs.clone()
            for ch in PHYSICAL_CHANNELS:
                inputs[:, :, mask_node, ch] = 0.0

        inputs_norm = inputs.clone()
        inputs_norm[..., :1] = scaler.transform(inputs[..., :1])
        inputs_norm = inputs_norm.to(device)
        out = model(inputs_norm, None, batch_seen=0, epoch=0, train=False)
        pred_raw = scaler.inverse_transform(out['prediction']).cpu()
        target_raw = target[..., :1]

        ae = torch.abs(pred_raw - target_raw)[..., 0]  # (B, T, N)
        all_ae.append(ae.mean(dim=1))  # (B, N)

    return torch.cat(all_ae, dim=0).numpy()  # (S, N)


@torch.no_grad()
def extract_spatial_attention(model, loader, scaler, device, n_batches=5):
    """Extract spatial attention weights from the model.

    Uses AttentionLayer.last_attn cache (set during forward pass).
    Returns: (N, N) averaged attention matrix — how much node i attends to node j.
    """
    attn_accum = torch.zeros(NUM_NODES, NUM_NODES, device=device)
    count = 0

    # model.py structure: model.spatial.attn_layers_s[0].attn is AttentionLayer
    # AttentionLayer stores self.last_attn = softmax(Q@K^T/sqrt(d)) after each forward
    spatial_attn_layer = model.spatial.attn_layers_s[0].attn

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        inputs = batch['inputs'].float()
        inputs_norm = inputs.clone()
        inputs_norm[..., :1] = scaler.transform(inputs[..., :1])
        inputs_norm = inputs_norm.to(device)

        # Full forward pass — spatial attention weights get cached in last_attn
        _ = model(inputs_norm, None, batch_seen=0, epoch=0, train=False)

        # last_attn: (num_heads*B, T, N, N)
        attn_weight = spatial_attn_layer.last_attn
        attn_avg = attn_weight.mean(dim=(0, 1))  # (N, N)
        attn_accum += attn_avg
        count += 1

    return (attn_accum / count).cpu().numpy()  # (N, N): row i = attention distribution of node i


def compute_input_stats(loader):
    """Compute per-node input statistics: mean, std, zero_rate for physical channels."""
    all_inputs = []
    for batch in loader:
        all_inputs.append(batch['inputs'].float())
    all_inputs = torch.cat(all_inputs, dim=0)  # (S, T, N, C)

    flow = all_inputs[..., 0]  # (S, T, N)
    occ = all_inputs[..., 1]
    speed = all_inputs[..., 2]

    # Per-node stats
    flow_mean = flow.mean(dim=(0, 1)).numpy()
    flow_std = flow.std(dim=(0, 1)).numpy()
    flow_zero_rate = (flow == 0).float().mean(dim=(0, 1)).numpy()

    return {
        'flow_mean': flow_mean,
        'flow_std': flow_std,
        'flow_zero_rate': flow_zero_rate,
    }


def print_section(title):
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-nodes', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    print(f'Device: {device}')

    model = load_model(device)
    adj = load_adj()
    categories = load_categories()
    loader, scaler = build_dataloader(args.n_samples)

    cat_order = [c for c in ['dead', 'major_fail', 'partial_fail', 'functional'] if c in categories]
    print(f'Categories: {[(k, len(v)) for k, v in categories.items()]}')

    rng = np.random.RandomState(42)
    sampled = {}
    for cat, indices in categories.items():
        n = min(args.n_nodes, len(indices))
        sampled[cat] = rng.choice(indices, n, replace=False)
    all_sampled = np.concatenate(list(sampled.values())).astype(int)

    # =====================================================================
    # Analysis 1: Baseline + single-node perturbation (top-K=10)
    # =====================================================================
    print_section('1. SINGLE-NODE PERTURBATION (top-K=10 neighbors)')

    neighbors_10 = get_topk_neighbors(adj, top_k=10)
    mae_clean = compute_per_node_mae(model, loader, scaler, device)
    print(f'Clean MAE: mean={mae_clean.mean():.3f}, median={np.median(mae_clean):.3f}')

    deltas = {}
    for v in tqdm(all_sampled, desc='Perturbation K=10'):
        v = int(v)
        nbrs = neighbors_10.get(v, np.array([]))
        if len(nbrs) == 0:
            deltas[v] = 0.0
            continue
        mae_masked = compute_per_node_mae(model, loader, scaler, device, mask_nodes=[v])
        deltas[v] = float((mae_masked[nbrs] - mae_clean[nbrs]).mean())

    print(f'\n{"Category":<15} {"n":>4} {"mean Δ":>10} {"std Δ":>10} {"min Δ":>10} {"max Δ":>10}')
    for cat in cat_order:
        if cat not in sampled:
            continue
        ds = [deltas[int(v)] for v in sampled[cat]]
        print(f'{cat:<15} {len(ds):>4} {np.mean(ds):>10.4f} {np.std(ds):>10.4f} '
              f'{np.min(ds):>10.4f} {np.max(ds):>10.4f}')

    # Effect size
    func_ds = [deltas[int(v)] for v in sampled.get('functional', [])]
    dead_ds = [deltas[int(v)] for v in sampled.get('dead', [])]
    if func_ds and dead_ds:
        tstat, pval = stats.mannwhitneyu(func_ds, dead_ds, alternative='greater')
        print(f'\nMann-Whitney U (functional > dead): U={tstat:.1f}, p={pval:.4f}')
        # Cohen's d
        pooled_std = np.sqrt((np.var(func_ds) + np.var(dead_ds)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(func_ds) - np.mean(dead_ds)) / pooled_std
            print(f"Cohen's d: {cohens_d:.3f}")

    # =====================================================================
    # Analysis 2: Top-K sensitivity
    # =====================================================================
    print_section('2. TOP-K SENSITIVITY (K=3, 5, 10, 20, 50)')

    # Pick 5 functional + 5 dead for speed
    func_5 = sampled.get('functional', [])[:5]
    dead_5 = sampled.get('dead', [])[:5]

    for K in [3, 5, 10, 20, 50]:
        neighbors_k = get_topk_neighbors(adj, top_k=K)
        func_deltas_k = []
        dead_deltas_k = []

        for v in func_5:
            v = int(v)
            nbrs = neighbors_k.get(v, np.array([]))
            if len(nbrs) == 0:
                continue
            mae_m = compute_per_node_mae(model, loader, scaler, device, mask_nodes=[v])
            func_deltas_k.append(float((mae_m[nbrs] - mae_clean[nbrs]).mean()))

        for v in dead_5:
            v = int(v)
            nbrs = neighbors_k.get(v, np.array([]))
            if len(nbrs) == 0:
                continue
            mae_m = compute_per_node_mae(model, loader, scaler, device, mask_nodes=[v])
            dead_deltas_k.append(float((mae_m[nbrs] - mae_clean[nbrs]).mean()))

        f_mean = np.mean(func_deltas_k) if func_deltas_k else 0
        d_mean = np.mean(dead_deltas_k) if dead_deltas_k else 0
        print(f'K={K:>3}: functional Δ={f_mean:>8.4f}, dead Δ={d_mean:>8.4f}, '
              f'gap={f_mean - d_mean:>8.4f}')

    # =====================================================================
    # Analysis 3: COLLECTIVE masking (all dead at once)
    # =====================================================================
    print_section('3. COLLECTIVE MASKING (all dead nodes at once)')

    dead_indices = categories['dead'].tolist()
    func_indices = categories['functional'].tolist()
    major_indices = categories['major_fail'].tolist()

    # Mask ALL dead nodes
    mae_no_dead = compute_per_node_mae(model, loader, scaler, device, mask_nodes=dead_indices)
    delta_no_dead_func = mae_no_dead[func_indices].mean() - mae_clean[func_indices].mean()
    print(f'Mask ALL {len(dead_indices)} dead nodes:')
    print(f'  Functional MAE: {mae_clean[func_indices].mean():.3f} → {mae_no_dead[func_indices].mean():.3f} (Δ={delta_no_dead_func:+.4f})')

    # Mask ALL major fail nodes
    mae_no_major = compute_per_node_mae(model, loader, scaler, device, mask_nodes=major_indices)
    delta_no_major_func = mae_no_major[func_indices].mean() - mae_clean[func_indices].mean()
    print(f'Mask ALL {len(major_indices)} major_fail nodes:')
    print(f'  Functional MAE: {mae_clean[func_indices].mean():.3f} → {mae_no_major[func_indices].mean():.3f} (Δ={delta_no_major_func:+.4f})')

    # Mask ALL dead + major
    mae_no_bad = compute_per_node_mae(model, loader, scaler, device,
                                       mask_nodes=dead_indices + major_indices)
    delta_no_bad_func = mae_no_bad[func_indices].mean() - mae_clean[func_indices].mean()
    print(f'Mask ALL {len(dead_indices)+len(major_indices)} dead+major nodes:')
    print(f'  Functional MAE: {mae_clean[func_indices].mean():.3f} → {mae_no_bad[func_indices].mean():.3f} (Δ={delta_no_bad_func:+.4f})')

    # Mask 20 random functional nodes (control)
    func_sample = rng.choice(func_indices, min(20, len(func_indices)), replace=False).tolist()
    remaining_func = [v for v in func_indices if v not in func_sample]
    mae_no_func20 = compute_per_node_mae(model, loader, scaler, device, mask_nodes=func_sample)
    delta_no_func20 = mae_no_func20[remaining_func].mean() - mae_clean[remaining_func].mean()
    print(f'Mask 20 random functional nodes (control):')
    print(f'  Remaining func MAE: {mae_clean[remaining_func].mean():.3f} → {mae_no_func20[remaining_func].mean():.3f} (Δ={delta_no_func20:+.4f})')

    # =====================================================================
    # Analysis 4: Spatial attention distribution
    # =====================================================================
    print_section('4. SPATIAL ATTENTION ANALYSIS')

    attn_matrix = extract_spatial_attention(model, loader, scaler, device)
    # attn_matrix[i, j] = how much node i attends to node j

    # How much attention do functional nodes receive vs dead nodes?
    attn_received = attn_matrix.mean(axis=0)  # (N,) — average attention each node receives
    print(f'Attention RECEIVED by category (mean across all nodes attending to them):')
    for cat in cat_order:
        indices = categories[cat].astype(int)
        print(f'  {cat:<15}: {attn_received[indices].mean():.6f} ± {attn_received[indices].std():.6f}')

    # Uniform baseline
    uniform = 1.0 / NUM_NODES
    print(f'  {"uniform":<15}: {uniform:.6f}')

    # Attention concentration: how much do functional nodes attend to dead?
    func_to_dead = attn_matrix[np.ix_(func_indices, dead_indices)].mean()
    func_to_func = attn_matrix[np.ix_(func_indices, func_indices)].mean()
    func_to_major = attn_matrix[np.ix_(func_indices, major_indices)].mean()
    print(f'\nFunctional nodes attend to:')
    print(f'  functional: {func_to_func:.6f} (expected if uniform: {len(func_indices)/NUM_NODES * uniform * NUM_NODES:.6f})')
    print(f'  dead:       {func_to_dead:.6f}')
    print(f'  major_fail: {func_to_major:.6f}')

    # Top-5 most attended nodes by functional nodes
    func_attn_to = attn_matrix[func_indices].mean(axis=0)  # avg attention from functional to each node
    top5_attended = np.argsort(func_attn_to)[-5:][::-1]
    print(f'\nTop-5 nodes most attended by functional:')
    for idx in top5_attended:
        cat_label = 'unknown'
        for c, inds in categories.items():
            if idx in inds:
                cat_label = c
                break
        print(f'  node {idx}: attn={func_attn_to[idx]:.6f} ({cat_label})')

    # =====================================================================
    # Analysis 5: Per-sample Δ variance (is signal consistent?)
    # =====================================================================
    print_section('5. PER-SAMPLE Δ VARIANCE (3 func + 3 dead)')

    func_3 = sampled.get('functional', [])[:3]
    dead_3 = sampled.get('dead', [])[:3]

    mae_clean_ps = compute_per_node_mae_per_sample(model, loader, scaler, device)  # (S, N)

    for label, nodes in [('functional', func_3), ('dead', dead_3)]:
        for v in nodes:
            v = int(v)
            nbrs = neighbors_10.get(v, np.array([]))
            if len(nbrs) == 0:
                continue
            mae_masked_ps = compute_per_node_mae_per_sample(model, loader, scaler, device, mask_node=v)
            # Per-sample Δ for this node's neighbors
            delta_ps = mae_masked_ps[:, nbrs] - mae_clean_ps[:, nbrs]  # (S, K)
            delta_ps_mean = delta_ps.mean(axis=1)  # (S,) — per-sample mean Δ
            pct_positive = (delta_ps_mean > 0).mean() * 100
            print(f'  node {v:>3} ({label:<12}): Δ mean={delta_ps_mean.mean():>8.4f}, '
                  f'std={delta_ps_mean.std():>8.4f}, '
                  f'>0: {pct_positive:.0f}%, '
                  f'p25={np.percentile(delta_ps_mean, 25):>8.4f}, '
                  f'p75={np.percentile(delta_ps_mean, 75):>8.4f}')

    # =====================================================================
    # Analysis 6: Input stats vs Δ correlation
    # =====================================================================
    print_section('6. INPUT STATS vs Δ CORRELATION')

    input_stats = compute_input_stats(loader)
    delta_arr = np.array([deltas[int(v)] for v in all_sampled])
    flow_mean_arr = input_stats['flow_mean'][all_sampled]
    flow_std_arr = input_stats['flow_std'][all_sampled]
    flow_zr_arr = input_stats['flow_zero_rate'][all_sampled]
    mae_arr = mae_clean[all_sampled]

    for name, arr in [('flow_mean', flow_mean_arr), ('flow_std', flow_std_arr),
                      ('flow_zero_rate', flow_zr_arr), ('clean_MAE', mae_arr)]:
        rho, pval = stats.spearmanr(arr, delta_arr)
        print(f'  Δ vs {name:<15}: ρ={rho:>7.3f} (p={pval:.3e})')

    # =====================================================================
    # Summary
    # =====================================================================
    print_section('SUMMARY')
    print(f'Single-node Δ (K=10):')
    for cat in cat_order:
        if cat not in sampled:
            continue
        ds = [deltas[int(v)] for v in sampled[cat]]
        print(f'  {cat:<15}: {np.mean(ds):+.4f} ± {np.std(ds):.4f}')

    print(f'\nCollective masking effect on functional MAE:')
    print(f'  Mask dead(120):        Δ={delta_no_dead_func:+.4f}')
    print(f'  Mask major(28):        Δ={delta_no_major_func:+.4f}')
    print(f'  Mask dead+major(148):  Δ={delta_no_bad_func:+.4f}')
    print(f'  Mask 20 functional:    Δ={delta_no_func20:+.4f}')

    print(f'\nAttention to dead (from functional): {func_to_dead:.6f} (uniform={uniform:.6f})')


if __name__ == '__main__':
    main()
