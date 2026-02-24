"""Case Study: Scale vs Pattern drift across years.

For selected nodes, visualize:
1. Raw daily profiles (shows scale change)
2. Normalized daily profiles (shows pattern preservation)
3. Scale trajectory across years
4. Quantitative decomposition

Goal: Visual motivation that "scale changes but pattern is preserved"
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("/data/pretrainingbasicts")

DATASETS = {
    2022: "datasets/SAN_BERNARDINO_2022_Q1",
    2023: "datasets/SAN_BERNARDINO_2023_Q1",
    2024: "datasets/SAN_BERNARDINO_2024_Q1",
}

STEPS_PER_DAY = 288  # 5-min intervals
OUTPUT_DIR = "eda/concept_drift/figures"


def load_data(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    return data


def compute_daily_profile(data, node_idx, channel=0):
    """Compute average daily profile for a node.
    Returns: (288,) average flow at each time-of-day.
    """
    flow = data[:, node_idx, channel]
    n_days = len(flow) // STEPS_PER_DAY
    # Reshape to (n_days, 288) and average
    daily = flow[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
    return daily.mean(axis=0), daily.std(axis=0), daily


def compute_node_stats(data, node_idx, channel=0):
    """Compute per-node statistics."""
    flow = data[:, node_idx, channel]
    return {
        "mean": float(np.mean(flow)),
        "std": float(np.std(flow)),
        "median": float(np.median(flow)),
        "max": float(np.max(flow)),
        "zero_rate": float(np.mean(flow == 0)),
    }


def select_representative_nodes(all_data):
    """Select nodes that best illustrate scale-pattern dynamics.

    IMPORTANT: Only consider nodes that are consistently functional
    across ALL three years (mean flow > 30 in every year).
    This filters out sensor state changes (dead/alive transitions).
    """
    MIN_FLOW_PER_YEAR = 50  # must have mean > 50 in EVERY year to be "consistently functional"
    MAX_SCALE_RATIO = 2.0   # >2x change is likely sensor failure, not genuine traffic drift

    # Compute per-node stats for each year (all nodes)
    node_stats = {}
    for year, data in all_data.items():
        for n in range(893):
            if n not in node_stats:
                node_stats[n] = {}
            flow = data[:, n, 0]
            node_stats[n][year] = {
                "mean": float(np.mean(flow)),
                "profile": compute_daily_profile(data, n)[0],
            }

    # Filter: consistently functional across all 3 years
    consistent = []
    for n in range(893):
        means = [node_stats[n][y]["mean"] for y in [2022, 2023, 2024]]
        if all(m >= MIN_FLOW_PER_YEAR for m in means):
            consistent.append(n)
    print(f"Consistently functional nodes (mean>{MIN_FLOW_PER_YEAR} in all years): {len(consistent)}/893")

    # For each consistent node, compute scale_change and pattern_corr
    candidates = []
    for n in consistent:
        means = [node_stats[n][y]["mean"] for y in [2022, 2023, 2024]]
        profiles = [node_stats[n][y]["profile"] for y in [2022, 2023, 2024]]

        scale_change = max(means) / min(means)

        # Skip likely sensor failures (>2x change is not genuine traffic drift)
        if scale_change > MAX_SCALE_RATIO:
            continue

        corrs = []
        for i in range(3):
            for j in range(i+1, 3):
                c = np.corrcoef(profiles[i], profiles[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        min_corr = min(corrs) if corrs else 0

        # Compute 2023 dip ratio (atmospheric river effect)
        dip_2023 = means[1] / means[0]  # <1 means 2023 had lower flow

        candidates.append({
            "node": n,
            "means": means,
            "scale_change": scale_change,
            "min_corr": min_corr,
            "mean_flow": np.mean(means),
            "dip_2023": dip_2023,
        })

    candidates.sort(key=lambda x: x["scale_change"], reverse=True)

    # Print distribution
    sc_arr = np.array([c["scale_change"] for c in candidates])
    print(f"Scale change distribution: median={np.median(sc_arr):.2f}, "
          f"mean={sc_arr.mean():.2f}, max={sc_arr.max():.2f}")

    # Select diverse representative nodes
    selected = {}
    used_nodes = set()

    def pick(name, condition):
        for c in candidates:
            if c["node"] not in used_nodes and condition(c):
                selected[name] = c
                used_nodes.add(c["node"])
                return True
        return False

    # 1. Clear 2023 dip + pattern preserved (best illustration of our hypothesis)
    #    Genuine traffic reduction: 10-30% dip, not sensor failure
    pick("2023_dip_pattern_stable",
         lambda c: 0.70 < c["dip_2023"] < 0.90 and c["min_corr"] > 0.95 and c["mean_flow"] > 100)

    # 2. Largest scale change with high pattern preservation
    pick("scale_change_high_corr",
         lambda c: c["scale_change"] > 1.15 and c["min_corr"] > 0.95 and c["mean_flow"] > 100)

    # 3. Stable scale + stable pattern (control/reference)
    for c in sorted(candidates, key=lambda x: x["scale_change"]):
        if c["node"] not in used_nodes and c["scale_change"] < 1.05 and c["min_corr"] > 0.97 and c["mean_flow"] > 100:
            selected["stable_control"] = c
            used_nodes.add(c["node"])
            break

    # 4. High flow node with 2023 dip
    pick("high_flow_2023_dip",
         lambda c: c["mean_flow"] > 250 and 0.75 < c["dip_2023"] < 0.92 and c["min_corr"] > 0.90)

    # 5. Moderate scale change + some pattern change (shows both drift types)
    pick("both_scale_and_pattern_change",
         lambda c: c["scale_change"] > 1.10 and 0.85 < c["min_corr"] < 0.94 and c["mean_flow"] > 80)

    return selected, node_stats, consistent


def plot_case_study(all_data, selected_nodes, node_stats, consistent=None):
    """Create comprehensive case study visualization."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    years = [2022, 2023, 2024]
    colors = {2022: '#2196F3', 2023: '#F44336', 2024: '#4CAF50'}
    year_labels = {2022: '2022 Q1', 2023: '2023 Q1 (Atm. River)', 2024: '2024 Q1'}
    time_hours = np.arange(288) / 12  # Convert to hours (288 steps / 12 steps per hour)

    # === Figure 1: Selected nodes - Raw vs Normalized profiles ===
    n_nodes = len(selected_nodes)
    fig, axes = plt.subplots(n_nodes, 2, figsize=(16, 4 * n_nodes), squeeze=False)
    fig.suptitle("Scale-Pattern Decomposition: Daily Traffic Profiles across Years",
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, (label, info) in enumerate(selected_nodes.items()):
        n = info["node"]
        means = info["means"]

        # Left: Raw daily profile
        ax1 = axes[idx, 0]
        for year in years:
            profile = node_stats[n][year]["profile"]
            ax1.plot(time_hours, profile, color=colors[year], label=year_labels[year],
                    linewidth=1.5, alpha=0.9)

        ax1.set_title(f"Node {n}: Raw Profile (Scale visible)\n"
                     f"Mean: {means[0]:.0f} → {means[1]:.0f} → {means[2]:.0f}  "
                     f"(Δ={info['scale_change']:.2f}x)",
                     fontsize=11)
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Flow (vehicles/5min)")
        ax1.legend(fontsize=9)
        ax1.set_xlim(0, 24)
        ax1.grid(True, alpha=0.3)

        # Right: Normalized daily profile (divide by daily mean)
        ax2 = axes[idx, 1]
        for year in years:
            profile = node_stats[n][year]["profile"]
            mean_val = np.mean(profile)
            if mean_val > 0:
                norm_profile = profile / mean_val
            else:
                norm_profile = profile
            ax2.plot(time_hours, norm_profile, color=colors[year], label=year_labels[year],
                    linewidth=1.5, alpha=0.9)

        ax2.set_title(f"Node {n}: Normalized Profile (Pattern visible)\n"
                     f"Min correlation: r={info['min_corr']:.3f}",
                     fontsize=11)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Normalized Flow (x / mean)")
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, 24)
        ax2.grid(True, alpha=0.3)

        # Add label on left
        ax1.annotate(label.replace("_", " ").title(),
                    xy=(-0.15, 0.5), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', rotation=90,
                    ha='center', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "daily_profiles_raw_vs_normalized.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: daily_profiles_raw_vs_normalized.png")

    # === Figure 2: Scale trajectory + Pattern correlation heatmap ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scale trajectory (mean flow per year)
    for label, info in selected_nodes.items():
        n = info["node"]
        means = info["means"]
        ax1.plot(years, means, 'o-', label=f"Node {n} ({label.replace('_', ' ')})",
                linewidth=2, markersize=8)

    ax1.set_title("Scale Trajectory: Per-Node Mean Flow", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean Flow (vehicles/5min)")
    ax1.set_xticks(years)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Pattern correlation matrix (aggregated over consistently functional nodes)
    corr_matrix = np.ones((3, 3))
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            if i < j:
                corrs = []
                for n in (consistent or []):
                    p1 = node_stats[n][y1]["profile"]
                    p2 = node_stats[n][y2]["profile"]
                    if np.std(p1) > 0 and np.std(p2) > 0:
                        c = np.corrcoef(p1, p2)[0, 1]
                        if not np.isnan(c):
                            corrs.append(c)
                corr_matrix[i, j] = np.mean(corrs)
                corr_matrix[j, i] = corr_matrix[i, j]

    im = ax2.imshow(corr_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0)
    ax2.set_title("Daily Profile Correlation\n(Functional Nodes, Mean)", fontsize=13, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([f"{y} Q1" for y in years])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([f"{y} Q1" for y in years])
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{corr_matrix[i,j]:.3f}", ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if corr_matrix[i,j] < 0.92 else 'black')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scale_trajectory_pattern_corr.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: scale_trajectory_pattern_corr.png")

    # === Figure 3: Distribution of scale change vs pattern correlation ===
    fig, ax = plt.subplots(figsize=(10, 7))

    scale_changes = []
    pattern_corrs = []
    mean_flows = []

    for n in (consistent or []):
        means = [node_stats[n][y]["mean"] for y in years]
        profiles = [node_stats[n][y]["profile"] for y in years]

        if min(means) < 1:
            continue

        sc = max(means) / min(means)

        corrs = []
        for i in range(3):
            for j in range(i+1, 3):
                c = np.corrcoef(profiles[i], profiles[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)

        if corrs:
            scale_changes.append(sc)
            pattern_corrs.append(min(corrs))
            mean_flows.append(np.mean(means))

    scatter = ax.scatter(scale_changes, pattern_corrs,
                        c=np.log10(mean_flows), cmap='viridis',
                        alpha=0.6, s=30, edgecolors='none')

    # Mark selected nodes
    for label, info in selected_nodes.items():
        n = info["node"]
        ax.scatter(info["scale_change"], info["min_corr"],
                  s=150, edgecolors='red', facecolors='none', linewidths=2,
                  zorder=5)
        ax.annotate(f"  Node {n}", (info["scale_change"], info["min_corr"]),
                   fontsize=9, fontweight='bold', color='red')

    ax.set_xlabel("Scale Change (max_mean / min_mean across years)", fontsize=12)
    ax.set_ylabel("Min Pattern Correlation (daily profile r)", fontsize=12)
    ax.set_title("Scale Change vs Pattern Preservation\n"
                "Each dot = one functional sensor node", fontsize=14, fontweight='bold')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='r=0.9')
    ax.axvline(x=1.2, color='gray', linestyle=':', alpha=0.5, label='1.2x scale change')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log10(Mean Flow)", fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scale_vs_pattern_scatter.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: scale_vs_pattern_scatter.png")

    # === Print summary statistics ===
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Functional Nodes)")
    print("="*70)

    sc_arr = np.array(scale_changes)
    pc_arr = np.array(pattern_corrs)

    print(f"\nScale Change (max/min mean across years):")
    print(f"  Mean: {sc_arr.mean():.3f}, Median: {np.median(sc_arr):.3f}")
    print(f"  >1.2x: {(sc_arr > 1.2).sum()}/{len(sc_arr)} ({(sc_arr > 1.2).mean()*100:.1f}%)")
    print(f"  >1.5x: {(sc_arr > 1.5).sum()}/{len(sc_arr)} ({(sc_arr > 1.5).mean()*100:.1f}%)")

    print(f"\nPattern Correlation (min pairwise profile r):")
    print(f"  Mean: {pc_arr.mean():.3f}, Median: {np.median(pc_arr):.3f}")
    print(f"  >0.95: {(pc_arr > 0.95).sum()}/{len(pc_arr)} ({(pc_arr > 0.95).mean()*100:.1f}%)")
    print(f"  >0.90: {(pc_arr > 0.90).sum()}/{len(pc_arr)} ({(pc_arr > 0.90).mean()*100:.1f}%)")

    # Key finding: nodes with large scale change but high pattern preservation
    both = (sc_arr > 1.2) & (pc_arr > 0.90)
    print(f"\n  Scale change >1.2x AND pattern r>0.90: {both.sum()}/{len(sc_arr)} "
          f"({both.mean()*100:.1f}%)")
    print(f"  → These nodes benefit most from disentanglement")

    # Selected nodes summary
    print(f"\n{'='*70}")
    print("SELECTED NODES")
    print(f"{'='*70}")
    for label, info in selected_nodes.items():
        n = info["node"]
        print(f"\n  [{label}] Node {n}:")
        print(f"    Mean flow: {info['means'][0]:.0f} → {info['means'][1]:.0f} → {info['means'][2]:.0f}")
        print(f"    Scale change: {info['scale_change']:.2f}x")
        print(f"    Min pattern corr: {info['min_corr']:.3f}")


def main():
    # Load all data
    all_data = {}
    for year, path in DATASETS.items():
        all_data[year] = load_data(path)
        print(f"Loaded {year}: shape={all_data[year].shape}")

    # Select representative nodes
    selected, node_stats, consistent = select_representative_nodes(all_data)

    print(f"\nSelected {len(selected)} representative nodes:")
    for label, info in selected.items():
        print(f"  {label}: Node {info['node']} "
              f"(scale={info['scale_change']:.2f}x, corr={info['min_corr']:.3f}, "
              f"mean={info['mean_flow']:.0f})")

    # Generate visualizations
    plot_case_study(all_data, selected, node_stats, consistent)


if __name__ == "__main__":
    main()
