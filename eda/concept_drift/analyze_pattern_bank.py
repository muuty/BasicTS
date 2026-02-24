"""Pattern Bank Prototype Analysis & Visualization.

Trains PB on specific cross-year pairs, then analyzes:
1. Node-prototype assignment (softmax weights) → spatial map
2. Prototype effect on predictions (what each prototype "means")
3. Correlation with node characteristics (flow, sensor health, type)
4. Adjacency-based spatial clustering
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from collections import defaultdict

sys.path.append("/data/pretrainingbasicts")
from baselines.STAEformer.arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893,
    "in_steps": 12,
    "out_steps": 12,
    "steps_per_day": 288,
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
}

INSTNORM_CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2022_Q1_30_12_12/ee3765011653c94baa0ab81045fb239e/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2023_Q1_30_12_12/d2940bc326d7f29eedeedb573e026973/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2024_Q1_30_12_12/16c8094c3cf1ef5013e0384708d2e954/STAEformer_best_val_MAE.pt",
}

DATASETS = {
    2022: "datasets/SAN_BERNARDINO_2022_Q1",
    2023: "datasets/SAN_BERNARDINO_2023_Q1",
    2024: "datasets/SAN_BERNARDINO_2024_Q1",
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
STEPS_PER_DAY = 288
DEVICE = "cuda:1"
FINETUNE_EPOCHS = 50
BATCH_SIZE = 16
NUM_PATTERNS = 8
LR = 0.003
OUTPUT_DIR = "eda/concept_drift/pb_analysis"


class STAEformerWithPatternBank(nn.Module):
    def __init__(self, backbone, num_nodes, adaptive_embedding_dim, num_patterns=8):
        super().__init__()
        self.backbone = backbone
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, adaptive_embedding_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, num_patterns))

    def get_adapter_output(self):
        weights = F.softmax(self.node_weights, dim=-1)
        return weights @ self.pattern_bank

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data
        batch_size = x.shape[0]
        enc = self.backbone.encoder

        if enc.tod_embedding_dim > 0:
            tod = x[..., enc.tod_index] * enc.steps_per_day
        if enc.dow_embedding_dim > 0:
            dow = x[..., enc.dow_index] * 7
        x = x[..., :enc.input_dim]

        x = enc.input_proj(x)
        features = [x]
        if enc.tod_embedding_dim > 0:
            features.append(enc.tod_embedding(tod.long()))
        if enc.dow_embedding_dim > 0:
            features.append(enc.dow_embedding(dow.long()))
        if enc.spatial_embedding_dim > 0:
            spatial_emb = enc.node_emb.expand(batch_size, enc.in_steps, *enc.node_emb.shape)
            features.append(spatial_emb)
        if enc.adaptive_embedding_dim > 0:
            adp_emb = enc.adaptive_embedding.expand(batch_size, *enc.adaptive_embedding.shape)
            adapter_out = self.get_adapter_output()
            adp_emb = adp_emb + adapter_out.unsqueeze(0).unsqueeze(0)
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)
        for attn in enc.attn_layers_t:
            x = attn(x, dim=1)
        x = self.backbone.spatial(x, None)
        out = self.backbone.decoder(x)
        return {"prediction": out}


def load_backbone(ckpt_path):
    backbone = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    return backbone


def load_data_and_scaler(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    train_data_ch0 = data[:n_train, :, 0]
    mean = float(np.mean(train_data_ch0))
    std = float(np.std(train_data_ch0))
    return data, mean, std, n_total


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+input_len+output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def normalize_input(x, mean, std):
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def apply_instance_norm(bx):
    flow = bx[:, :, :, 0]
    inst_mean = flow.mean(dim=1, keepdim=True)
    inst_std = flow.std(dim=1, keepdim=True) + 1e-5
    bx_normed = bx.clone()
    bx_normed[:, :, :, 0] = (flow - inst_mean) / inst_std
    return bx_normed, inst_mean, inst_std


def denorm_instance_norm(pred, inst_mean, inst_std):
    return pred * inst_std.unsqueeze(-1) + inst_mean.unsqueeze(-1)


def finetune_adapter(model, train_x, train_y, mean, std, epochs=10, lr=0.001):
    model = model.to(DEVICE)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = ("pattern_bank" in name or "node_weights" in name)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    train_x_norm = normalize_input(train_x, mean, std)
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_x_norm))
        epoch_loss, n_batches = 0, 0
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            bx = torch.FloatTensor(train_x_norm[batch_idx]).to(DEVICE)
            by = torch.FloatTensor(train_y[batch_idx]).to(DEVICE)
            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            pred = model(bx_normed, None, 0, 0, True)["prediction"]
            pred = denorm_instance_norm(pred, inst_mean, inst_std)
            pred_raw = pred * std + mean
            loss = nn.L1Loss()(pred_raw, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")
    model.eval()
    return model


def train_and_extract(source_year, target_year, n_hours, data_cache):
    """Train PB and extract learned parameters."""
    ckpt_path = INSTNORM_CHECKPOINTS[source_year]
    source_cache = data_cache[source_year]
    target_cache = data_cache[target_year]

    n_steps = n_hours * 12  # 5-min intervals
    ft_data = target_cache["full_data"][:n_steps]
    ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
    print(f"\nTraining PB: {source_year}->{target_year}, {n_hours}h ({len(ft_x)} samples)")

    backbone = load_backbone(ckpt_path)
    model = STAEformerWithPatternBank(
        backbone,
        num_nodes=MODEL_PARAM["num_nodes"],
        adaptive_embedding_dim=MODEL_PARAM["adaptive_embedding_dim"],
        num_patterns=NUM_PATTERNS,
    )
    model = finetune_adapter(model, ft_x, ft_y, source_cache["mean"], source_cache["std"],
                             epochs=FINETUNE_EPOCHS, lr=LR)

    # Extract learned parameters
    with torch.no_grad():
        pattern_bank = model.pattern_bank.cpu().numpy()  # (K, d)
        node_weights_raw = model.node_weights.cpu().numpy()  # (N, K)
        node_weights_soft = F.softmax(model.node_weights, dim=-1).cpu().numpy()  # (N, K)
        delta = model.get_adapter_output().cpu().numpy()  # (N, d)
        orig_emb = model.backbone.encoder.adaptive_embedding.cpu().numpy()  # (T, N, d)

    del model; torch.cuda.empty_cache()

    return {
        "pattern_bank": pattern_bank,
        "node_weights_raw": node_weights_raw,
        "node_weights_soft": node_weights_soft,
        "delta": delta,
        "orig_emb": orig_emb,
    }


def plot_spatial_map(coords_df, values, title, filename, cmap='tab10', categorical=True):
    """Plot sensor values on spatial map."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    lat = coords_df['Lat'].values
    lng = coords_df['Lng'].values

    if categorical:
        n_cats = len(np.unique(values))
        colors = plt.cm.get_cmap(cmap, n_cats)
        scatter = ax.scatter(lng, lat, c=values, cmap=colors, s=15, alpha=0.8,
                            edgecolors='k', linewidths=0.2)
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_cats))
        cbar.set_label('Prototype')
    else:
        scatter = ax.scatter(lng, lat, c=values, cmap=cmap, s=15, alpha=0.8,
                            edgecolors='k', linewidths=0.2)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(title.split(':')[-1].strip() if ':' in title else 'Value')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_weight_heatmap(weights, title, filename):
    """Plot node-prototype weight heatmap (subset)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Sort nodes by dominant prototype for cleaner visualization
    dominant = np.argmax(weights, axis=1)
    sort_idx = np.argsort(dominant)
    sorted_weights = weights[sort_idx]

    im = ax.imshow(sorted_weights, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Prototype Index')
    ax.set_ylabel('Node (sorted by dominant prototype)')
    ax.set_title(title)
    ax.set_xticks(range(weights.shape[1]))
    plt.colorbar(im, ax=ax, label='Softmax Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_prototype_vectors(pattern_bank, title, filename):
    """Visualize prototype vectors as heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    im = ax.imshow(pattern_bank, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Prototype')
    ax.set_yticks(range(pattern_bank.shape[0]))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_delta_magnitude_map(coords_df, delta, title, filename):
    """Plot magnitude of embedding change per node."""
    magnitude = np.linalg.norm(delta, axis=1)  # (N,)
    plot_spatial_map(coords_df, magnitude, title, filename, cmap='hot_r', categorical=False)


def analyze_prototype_characteristics(extracted, data_cache, source_year, target_year, coords_df):
    """Analyze what each prototype represents."""
    weights_soft = extracted["node_weights_soft"]  # (N, K)
    dominant_proto = np.argmax(weights_soft, axis=1)  # (N,)
    K = weights_soft.shape[1]

    # Node characteristics from source year data
    source_data = data_cache[source_year]["full_data"]
    target_data = data_cache[target_year]["full_data"]

    # Per-node stats from source
    n_train_src = int(source_data.shape[0] * TRAIN_RATIO)
    src_flow = source_data[:n_train_src, :, 0]  # (T, N)
    src_mean_flow = np.mean(src_flow, axis=0)  # (N,)
    src_std_flow = np.std(src_flow, axis=0)
    src_zero_rate = np.mean(src_flow == 0, axis=0)

    # Per-node stats from target
    n_train_tgt = int(target_data.shape[0] * TRAIN_RATIO)
    tgt_flow = target_data[:n_train_tgt, :, 0]
    tgt_mean_flow = np.mean(tgt_flow, axis=0)
    tgt_std_flow = np.std(tgt_flow, axis=0)

    # Flow change
    flow_change = tgt_mean_flow - src_mean_flow
    flow_change_pct = np.where(src_mean_flow > 1, flow_change / src_mean_flow * 100, 0)

    # Sensor type
    sensor_types = coords_df['Type'].values
    fwy = coords_df['Fwy'].values

    print(f"\n{'='*60}")
    print(f"PROTOTYPE CHARACTERISTICS: {source_year}->{target_year}")
    print(f"{'='*60}")

    proto_stats = []
    for k in range(K):
        mask = dominant_proto == k
        n_nodes = mask.sum()
        if n_nodes == 0:
            continue

        stats = {
            'proto': k,
            'n_nodes': int(n_nodes),
            'mean_flow_src': float(np.mean(src_mean_flow[mask])),
            'mean_flow_tgt': float(np.mean(tgt_mean_flow[mask])),
            'flow_change': float(np.mean(flow_change[mask])),
            'flow_change_pct': float(np.mean(flow_change_pct[mask])),
            'zero_rate': float(np.mean(src_zero_rate[mask])),
            'max_weight': float(np.mean(np.max(weights_soft[mask], axis=1))),
            'types': {},
            'fwy': {},
        }

        for t in np.unique(sensor_types[mask]):
            stats['types'][t] = int(np.sum(sensor_types[mask] == t))
        for f in np.unique(fwy[mask]):
            stats['fwy'][str(f)] = int(np.sum(fwy[mask] == f))

        proto_stats.append(stats)

        print(f"\n  Prototype {k}: {n_nodes} nodes")
        print(f"    Mean flow: {stats['mean_flow_src']:.1f} (src) -> {stats['mean_flow_tgt']:.1f} (tgt), "
              f"change: {stats['flow_change']:+.1f} ({stats['flow_change_pct']:+.1f}%)")
        print(f"    Zero rate: {stats['zero_rate']:.3f}")
        print(f"    Avg max weight: {stats['max_weight']:.3f}")
        print(f"    Types: {stats['types']}")
        top_fwy = sorted(stats['fwy'].items(), key=lambda x: -x[1])[:3]
        print(f"    Top freeways: {top_fwy}")

    return proto_stats


def plot_prototype_by_fwy(coords_df, dominant_proto, title, filename):
    """Plot prototype assignment colored by freeway."""
    K = len(np.unique(dominant_proto))
    fwys = coords_df['Fwy'].unique()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    lat = coords_df['Lat'].values
    lng = coords_df['Lng'].values

    colors_fwy = plt.cm.get_cmap('Set1', len(fwys))
    fwy_color_map = {f: colors_fwy(i) for i, f in enumerate(fwys)}

    for k in range(min(K, 8)):
        ax = axes[k]
        mask = dominant_proto == k

        # Plot all nodes in gray
        ax.scatter(lng[~mask], lat[~mask], c='lightgray', s=5, alpha=0.3)

        # Plot this prototype's nodes colored by freeway
        if mask.sum() > 0:
            node_colors = [fwy_color_map.get(f, 'black') for f in coords_df['Fwy'].values[mask]]
            ax.scatter(lng[mask], lat[mask], c=node_colors, s=20, alpha=0.9,
                      edgecolors='k', linewidths=0.3)

        ax.set_title(f'Proto {k} ({mask.sum()} nodes)')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=fwy_color_map[f], label=f'I-{f}') for f in fwys]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(fwys), fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_prototype_flow_change(proto_stats, title, filename):
    """Bar chart: flow change per prototype."""
    protos = [s['proto'] for s in proto_stats]
    changes = [s['flow_change'] for s in proto_stats]
    n_nodes = [s['n_nodes'] for s in proto_stats]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['red' if c < 0 else 'blue' for c in changes]
    bars = ax.bar(protos, changes, color=colors, alpha=0.7, edgecolor='k')

    # Add node count labels
    for bar, n in zip(bars, n_nodes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={n}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Prototype Index')
    ax.set_ylabel('Mean Flow Change (Target - Source)')
    ax.set_title(title)
    ax.set_xticks(protos)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_weight_entropy(weights_soft, coords_df, title, filename):
    """Plot entropy of mixing weights — low entropy = specialized, high = uniform."""
    entropy = -np.sum(weights_soft * np.log(weights_soft + 1e-10), axis=1)
    max_entropy = np.log(weights_soft.shape[1])
    normalized_entropy = entropy / max_entropy
    plot_spatial_map(coords_df, normalized_entropy, title, filename, cmap='coolwarm', categorical=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load coordinates
    coords_df = pd.read_csv('eda/concept_drift/san_bernardino_sensor_coords.csv', index_col=0)
    print(f"Loaded {len(coords_df)} sensor coordinates")

    # Load sensor categories
    dead_idx = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
    major_fail_idx = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')

    # Load data
    print("\nLoading data...")
    years = [2022, 2023, 2024]
    data_cache = {}
    for year in years:
        data, mean, std, n_total = load_data_and_scaler(DATASETS[year])
        data_cache[year] = {
            "full_data": data,
            "mean": mean,
            "std": std,
        }
        print(f"  {year}: mean={mean:.2f}, std={std:.2f}")

    # Analyze multiple cross-year pairs and adaptation durations
    pairs = [(2022, 2023), (2023, 2024), (2024, 2022)]
    hours_list = [12, 168]  # 12h (few-shot sweet spot) + 7d (for sharper prototypes)

    all_proto_stats = {}

    for source_year, target_year in pairs:
        for n_hours in hours_list:
            tag = f"{source_year}to{target_year}_{n_hours}h"
            print(f"\n{'='*60}")
            print(f"Analyzing: {source_year}->{target_year}, {n_hours}h")
            print(f"{'='*60}")

            extracted = train_and_extract(source_year, target_year, n_hours, data_cache)

            # 1. Prototype vectors heatmap
            plot_prototype_vectors(
                extracted["pattern_bank"],
                f"Learned Prototype Vectors ({source_year}->{target_year}, {n_hours}h)",
                f"proto_vectors_{tag}.png"
            )

            # 2. Node-prototype assignment map
            dominant_proto = np.argmax(extracted["node_weights_soft"], axis=1)
            plot_spatial_map(
                coords_df, dominant_proto,
                f"Dominant Prototype Assignment ({source_year}->{target_year}, {n_hours}h)",
                f"spatial_assignment_{tag}.png"
            )

            # 3. Per-prototype spatial breakdown by freeway
            plot_prototype_by_fwy(
                coords_df, dominant_proto,
                f"Prototype Spatial Distribution ({source_year}->{target_year}, {n_hours}h)",
                f"proto_by_fwy_{tag}.png"
            )

            # 4. Delta magnitude map
            plot_delta_magnitude_map(
                coords_df, extracted["delta"],
                f"Embedding Change Magnitude ({source_year}->{target_year}, {n_hours}h)",
                f"delta_magnitude_{tag}.png"
            )

            # 5. Weight heatmap
            plot_weight_heatmap(
                extracted["node_weights_soft"],
                f"Node-Prototype Weights ({source_year}->{target_year}, {n_hours}h)",
                f"weight_heatmap_{tag}.png"
            )

            # 6. Weight entropy map
            plot_weight_entropy(
                extracted["node_weights_soft"], coords_df,
                f"Mixing Weight Entropy ({source_year}->{target_year}, {n_hours}h)\nLow=Specialized, High=Uniform",
                f"entropy_map_{tag}.png"
            )

            # 7. Prototype characteristics analysis
            proto_stats = analyze_prototype_characteristics(
                extracted, data_cache, source_year, target_year, coords_df
            )
            all_proto_stats[tag] = proto_stats

            # 8. Flow change per prototype
            if proto_stats:
                plot_prototype_flow_change(
                    proto_stats,
                    f"Mean Flow Change by Prototype ({source_year}->{target_year}, {n_hours}h)",
                    f"flow_change_{tag}.png"
                )

            # 9. Mark dead/major_fail nodes
            sensor_health = np.zeros(893)
            sensor_health[dead_idx] = 2
            sensor_health[major_fail_idx] = 1
            # Check if dead nodes cluster in specific prototypes
            print(f"\n  Sensor health by prototype:")
            for k in range(NUM_PATTERNS):
                mask = dominant_proto == k
                n_dead = np.sum(sensor_health[mask] == 2)
                n_major = np.sum(sensor_health[mask] == 1)
                n_func = np.sum(sensor_health[mask] == 0)
                if mask.sum() > 0:
                    print(f"    Proto {k}: {mask.sum()} nodes "
                          f"(dead={n_dead}, major_fail={n_major}, functional={n_func})")

    # Save stats
    with open(os.path.join(OUTPUT_DIR, "prototype_stats.json"), "w") as f:
        json.dump(all_proto_stats, f, indent=2, default=str)
    print(f"\nAll analysis saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
