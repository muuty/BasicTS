"""Analyze gate values of STAEformerGated models.

Usage:
    python eda/analyze_gate.py <checkpoint_path> [--hidden_dim N] [--temperature T] [--bias B]

Examples:
    # Sharp gate (linear, tau=10, bias=-2)
    python eda/analyze_gate.py checkpoints/STAEformer_Gated_Sharp/SAN_BERNARDINO_30_12_12/*/STAEformerGated_best_val_MAE.pt --temperature 10 --bias -2

    # MLP gate
    python eda/analyze_gate.py checkpoints/STAEformer_Gated_MLP/SAN_BERNARDINO_30_12_12/*/STAEformerGated_best_val_MAE.pt --temperature 10 --bias -2 --hidden_dim 24

    # Default gate (tau=1, no bias)
    python eda/analyze_gate.py checkpoints/STAEformer_Gated/SAN_BERNARDINO_30_12_12/*/STAEformerGated_best_val_MAE.pt
"""
import argparse
import glob
import sys
sys.path.insert(0, '/data/pretrainingbasicts')

import torch
import numpy as np

from baselines.STAEformer.arch import STAEformerGated
from basicts.scaler import ZScoreScaler


def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def get_sensor_categories(data_range=(0, 26280)):
    data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r').reshape(105120, 893, 5)
    flow = data[data_range[0]:data_range[1], :, 0]
    zero_rate = (flow == 0).mean(axis=0)
    return {
        'dead': np.where(zero_rate > 0.9)[0],
        'major': np.where((zero_rate > 0.5) & (zero_rate <= 0.9))[0],
        'partial': np.where((zero_rate > 0.05) & (zero_rate <= 0.5))[0],
        'func': np.where(zero_rate <= 0.05)[0],
    }


def analyze_gate(ckpt_path, gate_temperature=1.0, gate_init_bias=0.0, gate_hidden_dim=0):
    # Load data
    data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r').reshape(105120, 893, 5)
    cats = get_sensor_categories()
    print(f"Categories: dead={len(cats['dead'])}, major={len(cats['major'])}, partial={len(cats['partial'])}, func={len(cats['func'])}")

    # Load model
    params = dict(
        num_nodes=893, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
        dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=24,
        feed_forward_dim=256, num_heads=4, num_layers=1, dropout=0.1, use_mixed_proj=True,
        gate_temperature=gate_temperature, gate_init_bias=gate_init_bias, gate_hidden_dim=gate_hidden_dim,
    )
    model = STAEformerGated(**params)
    sd = load_ckpt(ckpt_path)
    # Handle old key names
    sd = {k.replace('gate_linear.', 'gate_net.').replace('value_gate.0.', 'gate_net.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # Scaler
    scaler = ZScoreScaler(dataset_name='SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=False, rescale=True)

    # Hook to capture gate values
    captured_gates = {}
    def make_hook(name):
        def hook(module, input, output):
            value = input[2]
            gate = torch.sigmoid(module.gate_net(value) * module.gate_temperature)
            captured_gates[name] = gate.detach().cpu()
        return hook

    test_start = 21024
    steps_per_day = 288

    results = {}
    for time_label, hour in [('3am', 3), ('9am', 9), ('3pm', 15), ('9pm', 21)]:
        step_in_day = hour * 12
        indices = []
        for day_start in range(test_start, 26280 - 12, steps_per_day):
            idx = day_start + step_in_day
            if idx + 12 <= 26280:
                indices.append(idx)

        batch_raw = np.stack([data[i:i+12] for i in indices[:8]])
        batch_tensor = torch.tensor(batch_raw, dtype=torch.float32)
        batch_normed = scaler.transform(batch_tensor.clone())
        history_data = batch_normed[:, :, :, [0, 3, 4]]

        future_raw = np.stack([data[i+12:i+24] if i+24 <= 26280 else data[i:i+12] for i in indices[:8]])
        future_tensor = torch.tensor(future_raw, dtype=torch.float32)
        future_normed = scaler.transform(future_tensor.clone())
        future_data = future_normed[:, :, :, [0, 3, 4]]

        raw_flow = batch_raw[..., 0]

        handle = model.attn_layers_s[0].attn.register_forward_hook(make_hook(time_label))
        with torch.no_grad():
            model(history_data=history_data, future_data=future_data, batch_seen=0, epoch=0, train=False)
        handle.remove()

        gate_np = captured_gates[time_label].squeeze(-1).numpy()
        for cat_name, cat_idx in cats.items():
            g = gate_np[:, :, cat_idx]
            f = raw_flow[:, :, cat_idx]
            mask_zero = (f == 0)
            mask_pos = (f > 0)
            results[f"{time_label}_{cat_name}"] = {
                'mean': float(g.mean()),
                'f0': float(g[mask_zero].mean()) if mask_zero.any() else float('nan'),
                'fp': float(g[mask_pos].mean()) if mask_pos.any() else float('nan'),
            }

    # Print results
    print(f"\n=== Gate Values (temp={gate_temperature}, bias={gate_init_bias}, hidden={gate_hidden_dim}) ===")
    print(f"{'Time':<5} {'Cat':<10} {'Gate mean':>10} {'Flow=0':>10} {'Flow>0':>10}")
    print("-" * 50)
    for tl in ['3am', '9am', '3pm', '9pm']:
        for cat in ['dead', 'major', 'partial', 'func']:
            r = results[f"{tl}_{cat}"]
            print(f"{tl:<5} {cat:<10} {r['mean']:>10.4f} {r['f0']:>10.4f} {r['fp']:>10.4f}")
        print()

    # Time sensitivity
    print(f"{'Category':<10} {'3am':>8} {'9am':>8} {'3pm':>8} {'9pm':>8} {'Range':>8}")
    print("-" * 50)
    for cat in ['dead', 'major', 'partial', 'func']:
        vals = [results[f"{t}_{cat}"]['f0'] for t in ['3am', '9am', '3pm', '9pm']]
        rng = max(vals) - min(vals)
        print(f"{cat:<10} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f} {rng:>8.4f}")

    # Gate distribution
    all_g = np.concatenate([captured_gates[tl].squeeze(-1).numpy().flatten() for tl in ['3am', '9am', '3pm', '9pm']])
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.001]
    hist, _ = np.histogram(all_g, bins=bins)
    total = len(all_g)
    print(f"\nGate distribution:")
    for i in range(len(bins)-1):
        pct = hist[i] / total * 100
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {pct:>6.1f}%  {'#' * int(pct/2)}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', help='Path to checkpoint (supports glob)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=0)
    args = parser.parse_args()

    # Resolve glob
    paths = glob.glob(args.ckpt_path)
    if not paths:
        print(f"No checkpoint found: {args.ckpt_path}")
        sys.exit(1)
    ckpt = sorted(paths)[0]
    print(f"Checkpoint: {ckpt}")

    analyze_gate(ckpt, gate_temperature=args.temperature, gate_init_bias=args.bias, gate_hidden_dim=args.hidden_dim)
