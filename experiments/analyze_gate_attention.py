"""Analyze gate values by category and time of day for Sharp vs MLP gate models."""
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

# Load models
from baselines.STAEformer.arch import STAEformerGated

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

sharp_model = STAEformerGated(**{**params, 'gate_temperature': 10.0, 'gate_init_bias': -2.0, 'gate_hidden_dim': 0})
sharp_sd = load_ckpt('checkpoints/STAEformer_Gated_Sharp/SAN_BERNARDINO_30_12_12/535b8aa7d6dd926a0b72b539a4751d5e/STAEformerGated_best_val_MAE.pt')
# Remap old key name gate_linear -> gate_net
sharp_sd = {k.replace('gate_linear.', 'gate_net.'): v for k, v in sharp_sd.items()}
sharp_model.load_state_dict(sharp_sd)
sharp_model.eval()

mlp_model = STAEformerGated(**{**params, 'gate_temperature': 10.0, 'gate_init_bias': -2.0, 'gate_hidden_dim': 24})
mlp_sd = load_ckpt('checkpoints/STAEformer_Gated_MLP/SAN_BERNARDINO_30_12_12/5286b25259635796bb5a05a95890b25f/STAEformerGated_best_val_MAE.pt')
mlp_model.load_state_dict(mlp_sd)
mlp_model.eval()

# Scaler
from basicts.scaler import ZScoreScaler
scaler = ZScoreScaler(dataset_name='SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=False, rescale=True)

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

    batch_indices = indices[:8]
    batch = np.stack([data[i:i+12] for i in batch_indices])  # [8, 12, 893, 5]

    batch_norm = batch.copy()
    batch_norm[..., :3] = (batch_norm[..., :3] - scaler.mean.numpy()) / scaler.std.numpy()

    x = batch_norm[..., [0, 3, 4]]  # [8, 12, 893, 3]
    x_tensor = torch.tensor(x, dtype=torch.float32)
    raw_flow = batch[..., 0]  # [8, 12, 893]

    with torch.no_grad():
        for model_name, model in [('sharp', sharp_model), ('mlp', mlp_model)]:
            bx = x_tensor.clone()
            B = bx.shape[0]

            tod = bx[..., 1] * model.steps_per_day
            dow = bx[..., 2] * 7
            bx_input = bx[..., :model.input_dim]

            emb = model.input_proj(bx_input)
            features = [emb]
            if model.tod_embedding_dim > 0:
                features.append(model.tod_embedding(tod.long()))
            if model.dow_embedding_dim > 0:
                features.append(model.dow_embedding(dow.long()))
            if model.adaptive_embedding_dim > 0:
                features.append(model.adaptive_embedding.expand(B, *model.adaptive_embedding.shape))

            h = torch.cat(features, dim=-1)  # [B, T, N, 96]

            for attn in model.attn_layers_t:
                h = attn(h, dim=1)

            gate_layer = model.attn_layers_s[0].attn
            gate_val = torch.sigmoid(gate_layer.gate_net(h) * gate_layer.gate_temperature)
            gate_np = gate_val.squeeze(-1).numpy()  # [B, T, N]

            for cat_name, cat_idx in [('dead', dead), ('major', major_fail), ('partial', partial_fail), ('func', functional)]:
                g = gate_np[:, :, cat_idx]
                f = raw_flow[:, :, cat_idx]

                mask_zero = (f == 0)
                mask_pos = (f > 0)

                results[f"{model_name}_{time_label}_{cat_name}"] = {
                    'mean': float(g.mean()),
                    'f0': float(g[mask_zero].mean()) if mask_zero.any() else float('nan'),
                    'fp': float(g[mask_pos].mean()) if mask_pos.any() else float('nan'),
                    'pz': float(mask_zero.mean()),
                }

# Print table
print("\n=== Gate Values: Sharp (Linear) vs MLP ===")
print(f"{'Time':<5} {'Cat':<10} {'Sharp mean':>10} {'Sharp f=0':>10} {'Sharp f>0':>10} | {'MLP mean':>10} {'MLP f=0':>10} {'MLP f>0':>10}")
print("-" * 90)
for time_label in ['3am', '9am', '3pm', '9pm']:
    for cat in ['dead', 'major', 'partial', 'func']:
        s = results[f"sharp_{time_label}_{cat}"]
        m = results[f"mlp_{time_label}_{cat}"]
        print(f"{time_label:<5} {cat:<10} {s['mean']:>10.4f} {s['f0']:>10.4f} {s['fp']:>10.4f} | {m['mean']:>10.4f} {m['f0']:>10.4f} {m['fp']:>10.4f}")
    print()

# Key comparison: Does MLP gate vary by time for same category?
print("\n=== TIME SENSITIVITY: Gate(flow=0) across hours ===")
print(f"{'Category':<10} {'Model':<7} {'3am':>8} {'9am':>8} {'3pm':>8} {'9pm':>8} {'Range':>8}")
print("-" * 60)
for cat in ['dead', 'major', 'partial', 'func']:
    for mn in ['sharp', 'mlp']:
        vals = [results[f"{mn}_{t}_{cat}"]['f0'] for t in ['3am', '9am', '3pm', '9pm']]
        rng = max(vals) - min(vals)
        print(f"{cat:<10} {mn:<7} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f} {rng:>8.4f}")
    print()

# Gate distribution comparison
print("\n=== Gate Distribution (all test samples) ===")
# Collect all gate values across all hours
for mn, model in [('sharp', sharp_model), ('mlp', mlp_model)]:
    all_gates = []
    for time_label in ['3am', '9am', '3pm', '9pm']:
        for cat in ['dead', 'major', 'partial', 'func']:
            r = results[f"{mn}_{time_label}_{cat}"]
            # We need raw gate arrays; let's just report stats

    # Re-collect for histogram
    all_gate_vals = []
    step_in_day = 9 * 12  # use 9am as representative
    indices = []
    for day_start in range(test_start, 26280 - 12, steps_per_day):
        idx = day_start + step_in_day
        if idx + 12 <= 26280:
            indices.append(idx)
    batch = np.stack([data[i:i+12] for i in indices[:8]])
    batch_norm = batch.copy()
    batch_norm[..., :3] = (batch_norm[..., :3] - scaler.mean.numpy()) / scaler.std.numpy()
    x = batch_norm[..., [0, 3, 4]]
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        bx = x_tensor.clone()
        B = bx.shape[0]
        tod = bx[..., 1] * model.steps_per_day
        dow = bx[..., 2] * 7
        emb = model.input_proj(bx[..., :model.input_dim])
        features = [emb]
        features.append(model.tod_embedding(tod.long()))
        features.append(model.dow_embedding(dow.long()))
        features.append(model.adaptive_embedding.expand(B, *model.adaptive_embedding.shape))
        h = torch.cat(features, dim=-1)
        for attn in model.attn_layers_t:
            h = attn(h, dim=1)
        gate_layer = model.attn_layers_s[0].attn
        gate_val = torch.sigmoid(gate_layer.gate_net(h) * gate_layer.gate_temperature)
        g = gate_val.squeeze(-1).numpy().flatten()

    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.001]
    hist, _ = np.histogram(g, bins=bins)
    total = len(g)
    print(f"\n{mn.upper()} gate distribution:")
    for i in range(len(bins)-1):
        pct = hist[i] / total * 100
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {pct:>6.1f}%  {'#' * int(pct/2)}")
