"""Analyze gate values using actual model forward pass + hooks."""
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/pretrainingbasicts')

# Load data
data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r').reshape(105120, 893, 5)

# Sensor categories
data_slice = data[:26280]
flow = data_slice[:, :, 0]
zero_rate = (flow == 0).mean(axis=0)
dead = np.where(zero_rate > 0.9)[0]
major_fail = np.where((zero_rate > 0.5) & (zero_rate <= 0.9))[0]
partial_fail = np.where((zero_rate > 0.05) & (zero_rate <= 0.5))[0]
functional = np.where(zero_rate <= 0.05)[0]
print(f"Categories: dead={len(dead)}, major={len(major_fail)}, partial={len(partial_fail)}, functional={len(functional)}")

from baselines.STAEformer.arch import STAEformerGated
from basicts.scaler import ZScoreScaler

params = dict(
    num_nodes=893, in_steps=12, out_steps=12, steps_per_day=288,
    input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
    dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=24,
    feed_forward_dim=256, num_heads=4, num_layers=1, dropout=0.1, use_mixed_proj=True,
)

def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt

# Load sharp model
sharp_model = STAEformerGated(**{**params, 'gate_temperature': 10.0, 'gate_init_bias': -2.0, 'gate_hidden_dim': 0})
sharp_sd = load_ckpt('checkpoints/STAEformer_Gated_Sharp/SAN_BERNARDINO_30_12_12/535b8aa7d6dd926a0b72b539a4751d5e/STAEformerGated_best_val_MAE.pt')
# Check keys
print("\nSharp checkpoint keys (gate-related):")
for k in sorted(sharp_sd.keys()):
    if 'gate' in k:
        print(f"  {k}: shape={sharp_sd[k].shape}")

sharp_sd_remapped = {k.replace('gate_linear.', 'gate_net.'): v for k, v in sharp_sd.items()}
sharp_model.load_state_dict(sharp_sd_remapped)
sharp_model.eval()

# Load MLP model
mlp_model = STAEformerGated(**{**params, 'gate_temperature': 10.0, 'gate_init_bias': -2.0, 'gate_hidden_dim': 24})
mlp_sd = load_ckpt('checkpoints/STAEformer_Gated_MLP/SAN_BERNARDINO_30_12_12/5286b25259635796bb5a05a95890b25f/STAEformerGated_best_val_MAE.pt')
print("\nMLP checkpoint keys (gate-related):")
for k in sorted(mlp_sd.keys()):
    if 'gate' in k:
        print(f"  {k}: shape={mlp_sd[k].shape}")
mlp_model.load_state_dict(mlp_sd)
mlp_model.eval()

# Scaler
scaler = ZScoreScaler(dataset_name='SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=False, rescale=True)
print(f"\nScaler: mean={scaler.mean.item():.4f}, std={scaler.std.item():.4f}")

# Hook to capture gate values
captured_gates = {}

def make_gate_hook(name):
    def hook(module, input, output):
        # GatedAttentionLayer.forward receives (query, key, value)
        # value is input[2]
        value = input[2]
        gate = torch.sigmoid(module.gate_net(value) * module.gate_temperature)
        captured_gates[name] = gate.detach().cpu()
    return hook

# Prepare data exactly like the runner does
test_start = 21024
steps_per_day = 288

for time_label, hour in [('3am', 3), ('9am', 9), ('3pm', 15), ('9pm', 21)]:
    step_in_day = hour * 12
    indices = []
    for day_start in range(test_start, 26280 - 12, steps_per_day):
        idx = day_start + step_in_day
        if idx + 12 <= 26280:
            indices.append(idx)

    batch_indices = indices[:8]
    # Raw data: [8, 12, 893, 5]
    batch_raw = np.stack([data[i:i+12] for i in batch_indices])
    batch_tensor = torch.tensor(batch_raw, dtype=torch.float32)

    # Step 1: scaler.transform (normalizes channel 0 = flow only)
    batch_normed = scaler.transform(batch_tensor.clone())

    # Step 2: select forward features [0, 3, 4]
    history_data = batch_normed[:, :, :, [0, 3, 4]]  # [B, T, N, 3]

    # Also need future_data for model.forward (dummy)
    future_raw = np.stack([data[i+12:i+24] if i+24 <= 26280 else data[i:i+12] for i in batch_indices])
    future_tensor = torch.tensor(future_raw, dtype=torch.float32)
    future_normed = scaler.transform(future_tensor.clone())
    future_data = future_normed[:, :, :, [0, 3, 4]]

    raw_flow = batch_raw[..., 0]  # [8, 12, 893] for analysis

    for model_name, model in [('sharp', sharp_model), ('mlp', mlp_model)]:
        # Register hook
        gate_attn_layer = model.attn_layers_s[0].attn  # GatedAttentionLayer
        hook_handle = gate_attn_layer.register_forward_hook(make_gate_hook(f"{model_name}_{time_label}"))

        with torch.no_grad():
            model(history_data=history_data, future_data=future_data,
                  batch_seen=0, epoch=0, train=False)

        hook_handle.remove()

# Now analyze captured gates
print("\n=== Gate Values: Sharp (Linear) vs MLP ===")
print(f"{'Time':<5} {'Cat':<10} {'Sharp mean':>10} {'Sharp f=0':>10} {'Sharp f>0':>10} | {'MLP mean':>10} {'MLP f=0':>10} {'MLP f>0':>10}")
print("-" * 90)

results = {}
for time_label in ['3am', '9am', '3pm', '9pm']:
    step_in_day = int(time_label.replace('am','').replace('pm',''))
    if 'pm' in time_label and step_in_day != 12:
        step_in_day += 12
    hour = step_in_day

    indices = []
    for day_start in range(test_start, 26280 - 12, steps_per_day):
        idx = day_start + hour * 12
        if idx + 12 <= 26280:
            indices.append(idx)
    batch_raw = np.stack([data[i:i+12] for i in indices[:8]])
    raw_flow = batch_raw[..., 0]

    for model_name in ['sharp', 'mlp']:
        key = f"{model_name}_{time_label}"
        gate_np = captured_gates[key].squeeze(-1).numpy()  # [B, T, N]

        for cat_name, cat_idx in [('dead', dead), ('major', major_fail), ('partial', partial_fail), ('func', functional)]:
            g = gate_np[:, :, cat_idx]
            f = raw_flow[:, :, cat_idx]
            mask_zero = (f == 0)
            mask_pos = (f > 0)
            results[f"{key}_{cat_name}"] = {
                'mean': float(g.mean()),
                'f0': float(g[mask_zero].mean()) if mask_zero.any() else float('nan'),
                'fp': float(g[mask_pos].mean()) if mask_pos.any() else float('nan'),
            }

    for cat in ['dead', 'major', 'partial', 'func']:
        s = results[f"sharp_{time_label}_{cat}"]
        m = results[f"mlp_{time_label}_{cat}"]
        print(f"{time_label:<5} {cat:<10} {s['mean']:>10.4f} {s['f0']:>10.4f} {s['fp']:>10.4f} | {m['mean']:>10.4f} {m['f0']:>10.4f} {m['fp']:>10.4f}")
    print()

# Time sensitivity
print("\n=== TIME SENSITIVITY: Gate(flow=0) across hours ===")
print(f"{'Category':<10} {'Model':<7} {'3am':>8} {'9am':>8} {'3pm':>8} {'9pm':>8} {'Range':>8}")
print("-" * 60)
for cat in ['dead', 'major', 'partial', 'func']:
    for mn in ['sharp', 'mlp']:
        vals = [results[f"{mn}_{t}_{cat}"]['f0'] for t in ['3am', '9am', '3pm', '9pm']]
        rng = max(vals) - min(vals)
        print(f"{cat:<10} {mn:<7} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f} {rng:>8.4f}")
    print()

# Gate distribution
print("\n=== Gate Distribution ===")
for model_name in ['sharp', 'mlp']:
    all_g = []
    for tl in ['3am', '9am', '3pm', '9pm']:
        all_g.append(captured_gates[f"{model_name}_{tl}"].squeeze(-1).numpy().flatten())
    all_g = np.concatenate(all_g)
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.001]
    hist, _ = np.histogram(all_g, bins=bins)
    total = len(all_g)
    print(f"\n{model_name.upper()} gate distribution:")
    for i in range(len(bins)-1):
        pct = hist[i] / total * 100
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {pct:>6.1f}%  {'#' * int(pct/2)}")
