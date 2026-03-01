"""
Validate perturbation-based Δ as input quality signal.

Question: Does masking a node's input change NEIGHBOR predictions?
- Functional nodes: Δ > 0 (masking hurts neighbors)
- Dead nodes: Δ ≈ 0 (masking changes nothing)

Usage:
  python experiments/validate_delta.py --gpu 1
  python experiments/validate_delta.py --gpu 1 --n-samples 200 --n-nodes 10
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


def load_neighbors(top_k=10):
    """Load adjacency matrix and build top-K neighbor dict.

    The adj_mx is nearly fully connected (avg 847 neighbors).
    Using all neighbors dilutes the perturbation signal, so we
    take only the top-K strongest connections per node.
    """
    adj_path = f'datasets/xtraffic/{DATA_NAME}/adj_mx.pkl'
    with open(adj_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if isinstance(data, list) and len(data) >= 3:
        adj = data[2]
    else:
        adj = data
    adj = np.array(adj, dtype=np.float32)
    # Build top-K neighbor dict per node (by edge weight)
    neighbors = {}
    for v in range(adj.shape[0]):
        row = adj[v].copy()
        row[v] = 0.0  # exclude self
        topk_idx = np.argsort(row)[-top_k:]
        # Only keep those with positive weight
        topk_idx = topk_idx[row[topk_idx] > 0]
        neighbors[v] = topk_idx
    return neighbors, adj


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
        cats['partial_fail'] = np.array(sorted(all_nodes - set(dead) - set(major) - keep))
        cats['functional'] = np.array(sorted(keep))
    return cats


def build_dataloader(n_samples):
    rs = get_regular_settings(DATA_NAME)
    dataset = TimeSeriesForecastingDataset(
        DATA_NAME, rs['TRAIN_VAL_TEST_RATIO'], 'test',
        rs['INPUT_LEN'], rs['OUTPUT_LEN'], data_range=(0, 26280),
    )
    # Use subset for speed
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    scaler = ZScoreScaler(
        DATA_NAME, rs['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=rs['NORM_EACH_CHANNEL'], rescale=rs['RESCALE'],
    )
    return loader, scaler


@torch.no_grad()
def compute_per_node_mae(model, loader, scaler, device, mask_node=None):
    """Forward pass, return per-node MAE (N,) in original space.

    If mask_node is set, zero that node's physical channels before inference.
    """
    all_ae = []  # absolute errors per node

    for batch in loader:
        inputs = batch['inputs'].float()
        target = batch['target'].float()

        # Mask a node's physical channels if requested
        if mask_node is not None:
            inputs = inputs.clone()
            for ch in PHYSICAL_CHANNELS:
                inputs[:, :, mask_node, ch] = 0.0

        # Normalize flow (channel 0) via scaler
        # scaler.transform operates on [..., 0] internally, so pass the full tensor
        inputs_norm = inputs.clone()
        inputs_norm[..., :1] = scaler.transform(inputs[..., :1])  # (B,T,N,1) → keeps dim

        # Forward — model uses channels [0,1,2] (flow, occ, speed) as input_dim=3
        inputs_norm = inputs_norm.to(device)
        out = model(inputs_norm, None, batch_seen=0, epoch=0, train=False)
        pred_z = out['prediction']  # (B, T, N, 1) in z-score

        # Inverse transform — keep channel dim for scaler
        pred_raw = scaler.inverse_transform(pred_z).cpu()  # (B, T, N, 1)
        target_raw = target[..., :1]  # (B, T, N, 1) original scale

        ae = torch.abs(pred_raw - target_raw)  # (B, T, N, 1)
        ae = ae[..., 0]  # (B, T, N)
        all_ae.append(ae.mean(dim=1))  # mean over time → (B, N)

    all_ae = torch.cat(all_ae, dim=0)  # (S, N)
    return all_ae.mean(dim=0).numpy()  # (N,) mean over samples


def compute_gradient_utility(model, loader, scaler, device, n_batches=5):
    """Gradient-based utility: ||∂L/∂x_v|| for all nodes.

    Cheap: 1 forward + 1 backward per batch.
    """
    all_grad_norms = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        inputs = batch['inputs'].float()
        target = batch['target'].float()

        # Normalize
        inputs_norm = inputs.clone()
        inputs_norm[..., :1] = scaler.transform(inputs[..., :1])
        inputs_norm = inputs_norm.to(device)
        inputs_norm.requires_grad_(True)

        # Forward
        out = model(inputs_norm, None, batch_seen=0, epoch=0, train=False)
        pred_z = out['prediction'][..., 0]  # (B, T, N)

        # Target in z-score — keep channel dim for scaler
        target_z = scaler.transform(target[..., :1])[..., 0].to(device)

        # Loss = MAE over all nodes
        loss = torch.abs(pred_z - target_z).mean()
        loss.backward()

        # Gradient norm per node: mean over (B, T, C)
        grad = inputs_norm.grad  # (B, T, N, C)
        grad_per_node = grad.abs().mean(dim=(0, 1, 3))  # (N,)
        all_grad_norms.append(grad_per_node.detach().cpu())

        inputs_norm.requires_grad_(False)

    return torch.stack(all_grad_norms).mean(dim=0).numpy()  # (N,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-nodes', type=int, default=20,
                        help='Nodes to sample per category for perturbation')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    print(f'Device: {device}')

    # Load everything
    model = load_model(device)
    neighbors, adj = load_neighbors(top_k=10)
    categories = load_categories()
    loader, scaler = build_dataloader(args.n_samples)

    print(f'Test samples: {len(loader.dataset)}')
    print(f'Categories: {[(k, len(v)) for k, v in categories.items()]}')
    avg_neighbors = np.mean([len(v) for v in neighbors.values()])
    print(f'Avg neighbors per node: {avg_neighbors:.1f}')

    # Step 1: Clean baseline MAE
    print('\n=== Computing clean baseline MAE ===')
    mae_clean = compute_per_node_mae(model, loader, scaler, device)
    print(f'Clean MAE: mean={mae_clean.mean():.3f}, median={np.median(mae_clean):.3f}')

    # Step 2: Perturbation Δ for sampled nodes
    print(f'\n=== Computing perturbation Δ ({args.n_nodes} nodes per category) ===')
    rng = np.random.RandomState(42)

    sampled = {}
    for cat, indices in categories.items():
        n = min(args.n_nodes, len(indices))
        sampled[cat] = rng.choice(indices, n, replace=False)

    all_sampled = np.concatenate(list(sampled.values())).astype(int)
    deltas = {}

    for v in tqdm(all_sampled, desc='Perturbation'):
        v = int(v)
        nbrs = neighbors.get(v, np.array([]))
        if len(nbrs) == 0:
            deltas[v] = 0.0
            continue

        mae_masked = compute_per_node_mae(model, loader, scaler, device, mask_node=v)
        # Δ = how much masking v changes NEIGHBOR MAE
        delta_neighbors = mae_masked[nbrs] - mae_clean[nbrs]
        deltas[v] = float(delta_neighbors.mean())

    # Step 3: Gradient utility
    print('\n=== Computing gradient utility (all nodes) ===')
    grad_utility = compute_gradient_utility(model, loader, scaler, device)

    # Step 4: Analysis
    print('\n' + '='*60)
    print('RESULTS')
    print('='*60)

    print('\n--- Perturbation Δ by Category ---')
    print(f'{"Category":<15} {"n":>4} {"mean Δ":>10} {"std Δ":>10} {"Interpretation"}')
    cat_deltas = {}
    for cat in ['functional', 'partial_fail', 'major_fail', 'dead']:
        if cat not in sampled:
            continue
        ds = [deltas[int(v)] for v in sampled[cat]]
        cat_deltas[cat] = ds
        mean_d = np.mean(ds)
        std_d = np.std(ds)
        interp = 'USEFUL' if mean_d > 0.01 else ('HARMFUL' if mean_d < -0.01 else 'IRRELEVANT')
        print(f'{cat:<15} {len(ds):>4} {mean_d:>10.4f} {std_d:>10.4f} {interp}')

    print('\n--- Gradient Utility by Category ---')
    print(f'{"Category":<15} {"n":>4} {"mean grad":>10} {"std grad":>10}')
    for cat in ['functional', 'partial_fail', 'major_fail', 'dead']:
        if cat not in categories:
            continue
        indices = categories[cat].astype(int)
        g = grad_utility[indices]
        print(f'{cat:<15} {len(indices):>4} {g.mean():>10.6f} {g.std():>10.6f}')

    # Correlation between Δ and gradient utility
    print('\n--- Correlation: Δ vs Gradient Utility ---')
    delta_arr = np.array([deltas[int(v)] for v in all_sampled])
    grad_arr = np.array([grad_utility[int(v)] for v in all_sampled])
    rho, pval = stats.spearmanr(delta_arr, grad_arr)
    print(f'Spearman rho: {rho:.4f} (p={pval:.2e})')

    # Category ordering check
    print('\n--- Category Ordering Check ---')
    cat_order = ['dead', 'major_fail', 'partial_fail', 'functional']
    mean_deltas = [np.mean(cat_deltas.get(c, [0])) for c in cat_order]
    print(f'Expected: dead < major < partial < functional')
    print(f'Actual:   {" < ".join(f"{c}={d:.4f}" for c, d in zip(cat_order, mean_deltas))}')
    if mean_deltas == sorted(mean_deltas):
        print('PASS: ordering matches expectation')
    else:
        print('MIXED: ordering does not match perfectly')

    print('\n--- Verdict ---')
    func_delta = np.mean(cat_deltas.get('functional', [0]))
    dead_delta = np.mean(cat_deltas.get('dead', [0]))
    if func_delta > dead_delta and func_delta > 0:
        print('PROMISING: Δ distinguishes functional (useful) from dead (irrelevant)')
        print('→ Proceed with Δ-supervised σ training')
    else:
        print('INCONCLUSIVE: Δ does not clearly separate categories')
        print('→ Investigate further or try alternative approaches')


if __name__ == '__main__':
    main()
