"""Analyze spatial attention patterns of STAEformer models.

Captures attention scores from spatial self-attention layers and analyzes
how much attention functional nodes give to dead/major/partial/functional nodes.

Usage:
    python eda/analyze_attention.py <checkpoint_path> [--num_layers N] [--adaptive_dim D] [--gated] [--temperature T] [--bias B] [--hidden_dim H]

Examples:
    # Baseline STAEformer (1 layer, adaptive_dim=24)
    python eda/analyze_attention.py checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/*/STAEformer_Baseline_best_val_MAE.pt --num_layers 1 --adaptive_dim 24

    # Gated sharp model
    python eda/analyze_attention.py checkpoints/STAEformer_Gated_Sharp/SAN_BERNARDINO_30_12_12/*/STAEformerGated_best_val_MAE.pt --gated --temperature 10 --bias -2

    # Gated MLP model
    python eda/analyze_attention.py checkpoints/STAEformer_Gated_MLP/SAN_BERNARDINO_30_12_12/*/STAEformerGated_best_val_MAE.pt --gated --temperature 10 --bias -2 --hidden_dim 24

    # Gated soft model
    python eda/analyze_attention.py checkpoints/STAEformer_Gated/SAN_BERNARDINO_30_12_12/*/STAEformerGated_best_val_MAE.pt --gated
"""
import argparse
import glob
import sys
sys.path.insert(0, '/data/pretrainingbasicts')

import torch
import numpy as np

from baselines.STAEformer.arch.model import STAEformer as STAEformerModular
from baselines.STAEformer.arch.staeformer_arch import STAEformer as STAEformerMonolithic
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


def make_attn_hook(storage, name):
    """Hook that recomputes and captures attention scores from AttentionLayer."""
    def hook(module, inputs, output):
        query, key, value = inputs
        batch_size = query.shape[0]

        q = module.FC_Q(query)
        k = module.FC_K(key)

        # Split into heads: (num_heads * batch_size, ..., length, head_dim)
        q = torch.cat(torch.split(q, module.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(k, module.head_dim, dim=-1), dim=0)

        k_t = k.transpose(-1, -2)
        attn_score = (q @ k_t) / module.head_dim**0.5
        attn_score = torch.softmax(attn_score, dim=-1)

        # attn_score: (num_heads * batch_size, in_steps, num_nodes, num_nodes)
        # Average over heads: reshape to (num_heads, batch_size, in_steps, N, N) then mean over heads
        num_heads = module.num_heads
        attn_per_head = attn_score.reshape(num_heads, batch_size, *attn_score.shape[1:])
        attn_avg = attn_per_head.mean(dim=0)  # (batch_size, in_steps, N, N)

        storage[name] = attn_avg.detach().cpu()
    return hook


def analyze_attention(ckpt_path, num_layers=3, adaptive_dim=80,
                      gated=False, gate_temperature=1.0, gate_init_bias=0.0, gate_hidden_dim=0):
    data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r').reshape(105120, 893, 5)
    cats = get_sensor_categories()
    print(f"Categories: dead={len(cats['dead'])}, major={len(cats['major'])}, partial={len(cats['partial'])}, func={len(cats['func'])}")

    params = dict(
        num_nodes=893, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
        dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=adaptive_dim,
        feed_forward_dim=256, num_heads=4, num_layers=num_layers, dropout=0.1, use_mixed_proj=True,
    )
    sd = load_ckpt(ckpt_path)

    if gated:
        print(f"Gated model (temp={gate_temperature}, bias={gate_init_bias}, hidden={gate_hidden_dim})")
        model = STAEformerGated(**{**params,
            'gate_temperature': gate_temperature,
            'gate_init_bias': gate_init_bias,
            'gate_hidden_dim': gate_hidden_dim})
        sd = {k.replace('gate_linear.', 'gate_net.').replace('value_gate.0.', 'gate_net.'): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        # Detect architecture from checkpoint keys
        is_modular = any(k.startswith('encoder.') or k.startswith('spatial.') for k in sd.keys())
        if is_modular:
            print("Detected modular architecture (encoder/spatial/decoder)")
            model = STAEformerModular(**params)
        else:
            print("Detected monolithic architecture")
            model = STAEformerMonolithic(**params)
        model.load_state_dict(sd)
    model.eval()

    scaler = ZScoreScaler(dataset_name='SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=False, rescale=True)

    test_start = 21024
    steps_per_day = 288
    captured_attn = {}

    for time_label, hour in [('3am', 3), ('9am', 9), ('3pm', 15), ('9pm', 21)]:
        step_in_day = hour * 12
        indices = []
        for day_start in range(test_start, 26280 - 12, steps_per_day):
            idx = day_start + step_in_day
            if idx + 12 <= 26280:
                indices.append(idx)

        # Use small batch to save memory (attention matrix is N x N)
        batch_indices = indices[:4]
        batch_raw = np.stack([data[i:i+12] for i in batch_indices])
        batch_tensor = torch.tensor(batch_raw, dtype=torch.float32)
        batch_normed = scaler.transform(batch_tensor.clone())
        history_data = batch_normed[:, :, :, [0, 3, 4]]

        future_raw = np.stack([data[i+12:i+24] if i+24 <= 26280 else data[i:i+12] for i in batch_indices])
        future_tensor = torch.tensor(future_raw, dtype=torch.float32)
        future_normed = scaler.transform(future_tensor.clone())
        future_data = future_normed[:, :, :, [0, 3, 4]]

        # Hook on last spatial attention layer
        layer_idx = num_layers - 1
        if gated:
            attn_layer = model.attn_layers_s[layer_idx].attn
        elif is_modular:
            attn_layer = model.spatial.attn_layers_s[layer_idx].attn
        else:
            attn_layer = model.attn_layers_s[layer_idx].attn
        handle = attn_layer.register_forward_hook(
            make_attn_hook(captured_attn, time_label)
        )
        with torch.no_grad():
            model(history_data=history_data, future_data=future_data, batch_seen=0, epoch=0, train=False)
        handle.remove()

    # Analyze: For each source category, how much attention do they RECEIVE from each query category?
    # attn[i, j] = how much node i attends to node j (query=i, key=j)
    # We want: when functional nodes (query) attend, how much weight goes to dead nodes (key)?
    uniform = 1.0 / 893

    print(f"\n=== Attention Weights: How much does QUERY category attend to KEY category? ===")
    print(f"(Uniform attention = {uniform:.6f} per node)")
    print(f"\n{'Time':<5} {'Query':<10} {'-> dead':>12} {'-> major':>12} {'-> partial':>12} {'-> func':>12} | {'dead ratio':>10} {'func ratio':>10}")
    print("-" * 100)

    results = {}
    for tl in ['3am', '9am', '3pm', '9pm']:
        attn = captured_attn[tl].numpy()  # (B, T, N, N)
        # Average over batch and timesteps
        attn_mean = attn.mean(axis=(0, 1))  # (N, N) - attn_mean[i,j] = avg attention from i to j

        for q_cat_name, q_idx in cats.items():
            row = {}
            for k_cat_name, k_idx in cats.items():
                # Mean attention from q_cat nodes to k_cat nodes (per key node)
                attn_block = attn_mean[np.ix_(q_idx, k_idx)]  # (|q_cat|, |k_cat|)
                row[k_cat_name] = float(attn_block.mean())

            # Normalize: ratio vs uniform
            dead_ratio = row['dead'] / uniform
            func_ratio = row['func'] / uniform

            results[f"{tl}_{q_cat_name}"] = row
            print(f"{tl:<5} {q_cat_name:<10} {row['dead']:>12.6f} {row['major']:>12.6f} {row['partial']:>12.6f} {row['func']:>12.6f} | {dead_ratio:>10.3f}x {func_ratio:>10.3f}x")
        print()

    # Summary: Functional -> Dead attention vs Functional -> Functional attention
    print("=== SUMMARY: Functional nodes' attention distribution ===")
    print(f"{'Time':<5} {'Attn to dead':>15} {'Attn to func':>15} {'Ratio (func/dead)':>18}")
    print("-" * 55)
    for tl in ['3am', '9am', '3pm', '9pm']:
        d = results[f"{tl}_func"]['dead']
        f = results[f"{tl}_func"]['func']
        ratio = f / d if d > 0 else float('inf')
        print(f"{tl:<5} {d:>15.6f} {f:>15.6f} {ratio:>18.2f}x")

    # Per-node attention received (how much total attention each node receives)
    print(f"\n=== Per-node attention RECEIVED (column sum / N_query) ===")
    for tl in ['3am', '9am', '3pm', '9pm']:
        attn = captured_attn[tl].numpy().mean(axis=(0, 1))  # (N, N)
        # Column mean = average attention received by each key node from all query nodes
        col_mean = attn.mean(axis=0)  # (N,)
        print(f"\n{tl}:")
        for cat_name, cat_idx in cats.items():
            vals = col_mean[cat_idx]
            print(f"  {cat_name:<10}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")

    # Top and bottom attended nodes
    print(f"\n=== Top 10 LEAST attended nodes (averaged across times) ===")
    all_col_means = []
    for tl in ['3am', '9am', '3pm', '9pm']:
        attn = captured_attn[tl].numpy().mean(axis=(0, 1))
        all_col_means.append(attn.mean(axis=0))
    avg_received = np.mean(all_col_means, axis=0)

    bottom_10 = np.argsort(avg_received)[:10]
    flow_data = data[:26280, :, 0]
    zero_rates = (flow_data == 0).mean(axis=0)
    print(f"{'Rank':<5} {'Node':>5} {'Attn recv':>12} {'Zero rate':>10} {'Category':>10}")
    for i, idx in enumerate(bottom_10):
        zr = zero_rates[idx]
        cat = 'dead' if zr > 0.9 else ('major' if zr > 0.5 else ('partial' if zr > 0.05 else 'func'))
        print(f"{i+1:<5} {idx:>5} {avg_received[idx]:>12.6f} {zr:>10.3f} {cat:>10}")

    print(f"\n=== Top 10 MOST attended nodes ===")
    top_10 = np.argsort(avg_received)[-10:][::-1]
    for i, idx in enumerate(top_10):
        zr = zero_rates[idx]
        cat = 'dead' if zr > 0.9 else ('major' if zr > 0.5 else ('partial' if zr > 0.05 else 'func'))
        print(f"{i+1:<5} {idx:>5} {avg_received[idx]:>12.6f} {zr:>10.3f} {cat:>10}")

    return results, captured_attn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', help='Path to checkpoint (supports glob)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--adaptive_dim', type=int, default=80)
    parser.add_argument('--gated', action='store_true', help='Use STAEformerGated model')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=0)
    args = parser.parse_args()

    paths = glob.glob(args.ckpt_path)
    if not paths:
        print(f"No checkpoint found: {args.ckpt_path}")
        sys.exit(1)
    ckpt = sorted(paths)[0]
    print(f"Checkpoint: {ckpt}")

    analyze_attention(ckpt, num_layers=args.num_layers, adaptive_dim=args.adaptive_dim,
                      gated=args.gated, gate_temperature=args.temperature,
                      gate_init_bias=args.bias, gate_hidden_dim=args.hidden_dim)
