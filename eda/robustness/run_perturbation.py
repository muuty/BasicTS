"""
Test-time Perturbation Analysis: measure how corrupting n sensors affects
predictions of remaining healthy sensors.

Supports multiple models (STAEformer, STGCN) with identical experiment configs
for cross-model comparison.

Metrics computed per config:
  - System average: MAE degradation averaged over all healthy nodes
  - Top-K by Δ: degradation of nodes with largest prediction shift
  - Top-K by degradation: degradation of nodes with worst accuracy loss
  - Reach: count of nodes exceeding Δ or degradation thresholds

Outputs (saved to same directory as this script):
  {model}_clean_mae.npy              # (N_nodes,) per-node clean MAE
  {model}_per_node_delta.npy         # (n_configs, N_nodes) prediction shift
  {model}_per_node_degradation.npy   # (n_configs, N_nodes) MAE degradation
  {model}_summary.json               # full summary with top-K and reach stats

Usage:
  python eda/robustness/run_perturbation.py --model staeformer --gpu 1
  python eda/robustness/run_perturbation.py --model stgcn --gpu 1
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Experiment Config (shared across models) ────────────────────────
DATA_NAME = 'SAN_BERNARDINO'
DATASET_PATH = 'xtraffic/SAN_BERNARDINO'
NUM_NODES = 893
SEED = 42

CORRUPTION_CONFIGS = [
    # (type, intensity, label)
    ('flow_zero',  1.0, 'flow_zero'),
    ('all_zero',   1.0, 'all_zero'),
    ('noisy',      0.1, 'noisy_0.1'),
    ('noisy',      0.2, 'noisy_0.2'),
    ('noisy',      0.3, 'noisy_0.3'),
    ('spike',      0.5, 'spike_0.5'),
]

N_CORRUPT_OPTIONS = [5, 10, 20, 50]
TOP_K_VALUES = [5, 10, 20, 50]


# ─── Model Definitions ───────────────────────────────────────────────
def load_staeformer(device):
    from baselines.STAEformer.arch import STAEformer
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


def load_stgcn(device):
    from baselines.STGCN.arch import STGCN
    adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
    adj_mx = torch.Tensor(adj_mx[0])
    model = STGCN(
        Ks=3, Kt=3,
        blocks=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
        T=12, num_nodes=NUM_NODES,
        act_func='glu', graph_conv_type='cheb_graph_conv',
        adj_matrix=adj_mx, bias=True, droprate=0.5,
    )
    ckpt = 'checkpoints/STGCN/SAN_BERNARDINO_30_12_12/70d4d891bd58bc747849b30bd86f9774/STGCN_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_agcrn(device):
    from baselines.AGCRN.arch import AGCRN
    model = AGCRN(
        num_nodes=NUM_NODES, input_dim=1, rnn_units=64, output_dim=1,
        horizon=12, num_layers=2, default_graph=True, embed_dim=10, cheb_k=2,
    )
    import glob
    ckpt_dir = 'checkpoints/AGCRN/SAN_BERNARDINO_30_12_12/'
    candidates = glob.glob(os.path.join(ckpt_dir, '*/AGCRN_best_val_MAE.pt'))
    ckpt = candidates[0]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


MODEL_REGISTRY = {
    'staeformer': {
        'loader': load_staeformer,
        'forward_features': [0, 1, 2, 3, 4],  # flow, occ, speed, tod, dow
        'has_future': True,                     # needs tod/dow in future
    },
    'stgcn': {
        'loader': load_stgcn,
        'forward_features': [0],                # flow only
        'has_future': False,
    },
    'agcrn': {
        'loader': load_agcrn,
        'forward_features': [0],                # flow only
        'has_future': False,
    },
}


# ─── Corruption ──────────────────────────────────────────────────────
def apply_corruption(inputs, corrupt_nodes, ctype, intensity, flow_std, rng):
    """Apply corruption to raw (un-normalized) inputs. Only modifies flow (ch0)."""
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]
    if ctype == 'flow_zero':
        corrupted[:, :, corrupt_nodes, 0] = 0.0
    elif ctype == 'all_zero':
        for ch in [0, 1, 2]:
            corrupted[:, :, corrupt_nodes, ch] = 0.0
    elif ctype == 'noisy':
        noise = torch.tensor(
            rng.normal(0, intensity * flow_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device)
        corrupted[:, :, corrupt_nodes, 0] += noise
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)
    elif ctype == 'spike':
        spike_mask = torch.tensor(
            rng.random((B, T, len(corrupt_nodes))) < 0.5,
            dtype=corrupted.dtype, device=corrupted.device)
        spike_vals = torch.tensor(
            rng.normal(0, intensity * flow_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device)
        corrupted[:, :, corrupt_nodes, 0] += spike_mask * spike_vals
        corrupted[:, :, corrupt_nodes, 0].clamp_(min=0)
    return corrupted


# ─── Inference ────────────────────────────────────────────────────────
def run_inference(model, inputs_norm, device, model_cfg):
    """Run model forward pass, return predictions (B, 12, N) in normalized space."""
    fwd = model_cfg['forward_features']
    history = inputs_norm[..., fwd].to(device)

    if model_cfg['has_future']:
        future = torch.zeros_like(history)
        future[..., 3] = history[:, -1:, :, 3].expand_as(future[..., 3])
        future[..., 4] = history[:, -1:, :, 4].expand_as(future[..., 4])
    else:
        future = None

    with torch.no_grad():
        pred = model(history_data=history, future_data=future,
                     batch_seen=0, epoch=0, train=False)
    if isinstance(pred, dict):
        pred = pred['prediction']
    if pred.dim() == 4:
        pred = pred[..., 0]
    return pred.cpu()


# ─── Analysis ─────────────────────────────────────────────────────────
def compute_per_node_masked_mae(pred, targets, mask):
    mae = np.zeros(pred.shape[2])
    for n in range(pred.shape[2]):
        m = mask[:, :, n]
        if m.sum() > 0:
            mae[n] = np.abs(pred[:, :, n][m] - targets[:, :, n][m]).mean()
    return mae


def top_k_stats(values, k):
    sorted_vals = np.sort(values)[::-1]
    top = sorted_vals[:k]
    return {'mean': float(top.mean()), 'max': float(top.max()), 'min': float(top.min())}


def analyze_config(pred_clean, pred_corrupted, targets, mask,
                   clean_mae, corrupt_nodes, functional_nodes):
    healthy = np.setdiff1d(functional_nodes, corrupt_nodes)
    corrupted_mae = compute_per_node_masked_mae(pred_corrupted, targets, mask)
    delta_per_node = np.abs(pred_corrupted - pred_clean).mean(axis=(0, 1))
    degradation_per_node = corrupted_mae - clean_mae

    h_delta = delta_per_node[healthy]
    h_degrad = degradation_per_node[healthy]

    sort_by_delta = np.argsort(-h_delta)
    sort_by_degrad = np.argsort(-h_degrad)

    result = {
        'n_healthy': len(healthy),
        'system_avg_delta': float(h_delta.mean()),
        'system_avg_degradation': float(h_degrad.mean()),
        'corrupt_avg_delta': float(delta_per_node[corrupt_nodes].mean()),
        'corrupt_avg_degradation': float(degradation_per_node[corrupt_nodes].mean()),
    }

    for k in TOP_K_VALUES:
        if k > len(sort_by_delta):
            continue
        idx_d = sort_by_delta[:k]
        result[f'by_delta_top{k}_delta'] = top_k_stats(h_delta[idx_d], k)
        result[f'by_delta_top{k}_degradation'] = top_k_stats(h_degrad[idx_d], k)
        idx_g = sort_by_degrad[:k]
        result[f'by_degrad_top{k}_delta'] = top_k_stats(h_delta[idx_g], k)
        result[f'by_degrad_top{k}_degradation'] = top_k_stats(h_degrad[idx_g], k)

    for t in [0.1, 0.5, 1.0, 2.0, 5.0]:
        result[f'reach_delta_gt_{t}'] = int((h_delta > t).sum())
        result[f'reach_degrad_gt_{t}'] = int((h_degrad > t).sum())

    return result, delta_per_node, degradation_per_node


# ─── Print Tables ─────────────────────────────────────────────────────
def print_tables(summary, model_name):
    clean_avg = summary['clean_mae_functional_avg']

    for table_id, sort_key, sort_label in [
        ('A', 'by_delta', 'sorted by Δ (prediction shift)'),
        ('B', 'by_degrad', 'sorted by degradation (worst accuracy)'),
    ]:
        print(f"\n{'=' * 130}")
        print(f"{model_name} — TABLE {table_id}: Top-K {sort_label} — MAE degradation (Clean MAE = {clean_avg:.4f})")
        print(f"{'=' * 130}")
        print(f"{'Config':<20} {'n':>4} {'SysAvg':>10} | {'Top5':>10} {'Top10':>10} {'Top20':>10} {'Top50':>10} | {'Reach>0.5':>10}")
        print("-" * 130)
        for ctype, intensity, label in CORRUPTION_CONFIGS:
            for nc in N_CORRUPT_OPTIONS:
                key = f"{label}_n{nc}"
                r = summary['configs'][key]
                sys_d = r['system_avg_degradation']
                vals = [r.get(f'{sort_key}_top{k}_degradation', {}).get('mean', float('nan'))
                        for k in TOP_K_VALUES]
                reach_key = 'reach_delta_gt_0.5' if sort_key == 'by_delta' else 'reach_degrad_gt_0.5'
                reach = r.get(reach_key, 0)
                print(f"  {label:<18} {nc:>4} {sys_d:>+10.3f} | "
                      f"{vals[0]:>+10.3f} {vals[1]:>+10.3f} {vals[2]:>+10.3f} {vals[3]:>+10.3f} | "
                      f"{reach:>10}")
            print()

    print(f"{'=' * 130}")
    print(f"{model_name} — TABLE C: Top-K Δ values (sorted by Δ)")
    print(f"{'=' * 130}")
    print(f"{'Config':<20} {'n':>4} {'SysAvg':>10} | {'Top5':>10} {'Top10':>10} {'Top20':>10} {'Top50':>10}")
    print("-" * 130)
    for ctype, intensity, label in CORRUPTION_CONFIGS:
        for nc in N_CORRUPT_OPTIONS:
            key = f"{label}_n{nc}"
            r = summary['configs'][key]
            sys_d = r['system_avg_delta']
            vals = [r.get(f'by_delta_top{k}_delta', {}).get('mean', float('nan'))
                    for k in TOP_K_VALUES]
            print(f"  {label:<18} {nc:>4} {sys_d:>10.3f} | "
                  f"{vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {vals[3]:>10.3f}")
        print()


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test-time perturbation analysis')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model to analyze')
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()

    model_name = args.model
    model_cfg = MODEL_REGISTRY[model_name]
    device = torch.device(f'cuda:{args.gpu}')

    print(f"=== {model_name.upper()} perturbation analysis ===")
    print(f"Loading model...")
    model = model_cfg['loader'](device)

    print("Loading scaler and dataset...")
    rs = get_regular_settings(DATA_NAME)
    scaler = ZScoreScaler(
        dataset_name=DATASET_PATH,
        train_ratio=rs['TRAIN_VAL_TEST_RATIO'][0],
        norm_each_channel=rs['NORM_EACH_CHANNEL'],
        rescale=rs['RESCALE'],
    )
    dataset = TimeSeriesForecastingDataset(
        dataset_name=DATASET_PATH,
        train_val_test_ratio=rs['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12,
        data_range=(0, 26280),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    mean = scaler.mean.float()
    std = scaler.std.float()
    flow_std = std.item() if std.dim() == 0 else std.mean().item()

    dead = np.load(f'datasets/{DATASET_PATH}/dead_indices.npy')
    major_fail = np.load(f'datasets/{DATASET_PATH}/major_fail_indices.npy')
    functional = np.setdiff1d(np.arange(NUM_NODES), np.union1d(dead, major_fail))
    print(f"  Functional: {len(functional)}, Dead: {len(dead)}, Major: {len(major_fail)}")

    # ── Clean baseline ──
    print("Computing clean predictions...")
    all_pred, all_targets = [], []
    for batch in loader:
        inputs_raw = batch['inputs'].float()
        target_raw = batch['target'].float()
        inputs_norm = inputs_raw.clone()
        inputs_norm[..., 0] = (inputs_norm[..., 0] - mean) / std
        pred_norm = run_inference(model, inputs_norm, device, model_cfg)
        all_pred.append((pred_norm * std + mean).numpy())
        all_targets.append(target_raw[..., 0].numpy())

    pred_clean = np.concatenate(all_pred, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mask = targets > 0
    clean_mae = compute_per_node_masked_mae(pred_clean, targets, mask)
    print(f"  Clean MAE (functional avg): {clean_mae[functional].mean():.4f}")

    # ── Perturbation experiments ──
    config_labels = []
    all_delta, all_degrad = [], []
    summary = {
        'model': model_name,
        'clean_mae_functional_avg': float(clean_mae[functional].mean()),
        'n_functional': len(functional),
        'forward_features': model_cfg['forward_features'],
        'configs': {},
    }

    total = len(CORRUPTION_CONFIGS) * len(N_CORRUPT_OPTIONS)
    done = 0

    for ctype, intensity, label in CORRUPTION_CONFIGS:
        for n_corrupt in N_CORRUPT_OPTIONS:
            config_key = f"{label}_n{n_corrupt}"
            done += 1
            print(f"[{done}/{total}] {config_key}...")

            rng_select = np.random.RandomState(SEED)
            corrupt_nodes = np.sort(rng_select.choice(
                functional, min(n_corrupt, len(functional)), replace=False))

            rng_corrupt = np.random.RandomState(SEED + 1000)
            all_pred_corrupted = []
            for batch in loader:
                inputs_raw = batch['inputs'].float()
                inputs_corrupted = apply_corruption(
                    inputs_raw, corrupt_nodes, ctype, intensity, flow_std, rng_corrupt)
                inputs_corrupted[..., 0] = (inputs_corrupted[..., 0] - mean) / std
                pred_norm = run_inference(model, inputs_corrupted, device, model_cfg)
                all_pred_corrupted.append((pred_norm * std + mean).numpy())
            pred_corrupted = np.concatenate(all_pred_corrupted, axis=0)

            result, delta, degrad = analyze_config(
                pred_clean, pred_corrupted, targets, mask,
                clean_mae, corrupt_nodes, functional)

            result['corruption_type'] = ctype
            result['intensity'] = intensity
            result['n_corrupt'] = n_corrupt
            result['corrupt_node_indices'] = corrupt_nodes.tolist()

            summary['configs'][config_key] = result
            config_labels.append(config_key)
            all_delta.append(delta)
            all_degrad.append(degrad)

            sys_d = result['system_avg_degradation']
            bd5 = result.get('by_delta_top5_degradation', {}).get('mean', 0)
            bg5 = result.get('by_degrad_top5_degradation', {}).get('mean', 0)
            print(f"  sys_deg={sys_d:+.3f}  by_Δ_top5={bd5:+.3f}  by_deg_top5={bg5:+.3f}")

    # ── Save ──
    prefix = model_name
    np.save(os.path.join(OUTPUT_DIR, f'{prefix}_clean_mae.npy'), clean_mae)
    all_delta = np.stack(all_delta, axis=0)
    all_degrad = np.stack(all_degrad, axis=0)
    np.save(os.path.join(OUTPUT_DIR, f'{prefix}_per_node_delta.npy'), all_delta)
    np.save(os.path.join(OUTPUT_DIR, f'{prefix}_per_node_degradation.npy'), all_degrad)

    summary['config_labels'] = config_labels
    with open(os.path.join(OUTPUT_DIR, f'{prefix}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print ──
    print_tables(summary, model_name.upper())

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  {prefix}_clean_mae.npy              ({NUM_NODES},)")
    print(f"  {prefix}_per_node_delta.npy          ({len(config_labels)}, {NUM_NODES})")
    print(f"  {prefix}_per_node_degradation.npy    ({len(config_labels)}, {NUM_NODES})")
    print(f"  {prefix}_summary.json")


if __name__ == '__main__':
    main()
