"""Detailed concept drift analysis: per-node, per-ToD, per-DoW breakdown.

Questions:
1. Is drift concentrated in specific nodes, or gradual across all?
2. Which nodes/tod/dow show the largest degradation?
3. What behavioral insights are hidden in the degradation patterns?
"""
import sys
import os
import json
import numpy as np
import torch
sys.path.append("/data/pretrainingbasicts")

from baselines.STAEformer.arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893, "in_steps": 12, "out_steps": 12, "steps_per_day": 288,
    "input_dim": 3, "output_dim": 1, "input_embedding_dim": 24,
    "tod_embedding_dim": 24, "dow_embedding_dim": 24, "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24, "feed_forward_dim": 256, "num_heads": 4,
    "num_layers": 1, "dropout": 0.1, "use_mixed_proj": True,
}

CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2022_Q1_30_12_12/6d33ff60f58f5fa9e14b8d42bfdda7a8/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2023_Q1_30_12_12/0de30023c9399c9d238c0c7bbbfba3d6/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2024_Q1_30_12_12/a00dabf5d051255d7be14e2acb8c7945/STAEformer_best_val_MAE.pt",
}

DATASETS = {
    2022: "datasets/SAN_BERNARDINO_2022_Q1",
    2023: "datasets/SAN_BERNARDINO_2023_Q1",
    2024: "datasets/SAN_BERNARDINO_2024_Q1",
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
DEVICE = "cuda:1"
NUM_NODES = 893
STEPS_PER_DAY = 288
OUTPUT_DIR = "eda/concept_drift"


def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


def load_data_and_scaler(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    train_data_ch0 = data[:n_train, :, 0]
    mean = float(np.mean(train_data_ch0))
    std = float(np.std(train_data_ch0))
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val
    test_data = data[test_start:]
    return test_data, mean, std


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    # Also track tod/dow of prediction target
    samples_tod, samples_dow = [], []
    total_len = input_len + output_len
    for i in range(len(data) - total_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+total_len, :, 0:1])
        # tod/dow from the first prediction step
        samples_tod.append(data[i+input_len, 0, 3])  # tod channel, any node (same for all)
        samples_dow.append(data[i+input_len, 0, 4])  # dow channel
    return np.array(samples_x), np.array(samples_y), np.array(samples_tod), np.array(samples_dow)


def predict(model, test_x, mean, std, batch_size=64):
    test_x_norm = test_x.copy()
    test_x_norm[:, :, :, 0] = (test_x[:, :, :, 0] - mean) / std
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    return np.concatenate(all_preds, axis=0)


def analyze_pair(train_year, test_year, model, test_x, test_y, test_tod, test_dow, mean, std):
    """Full analysis for one (train_year, test_year) pair."""
    preds = predict(model, test_x, mean, std)  # (N, 12, 893, 1)
    errors = np.abs(preds - test_y)  # (N, 12, 893, 1)
    errors = errors.squeeze(-1)  # (N, 12, 893)

    result = {}

    # 1. Per-node MAE: (893,)
    per_node_mae = errors.mean(axis=(0, 1))  # avg over samples and horizons
    result["per_node_mae"] = per_node_mae

    # 2. Per-ToD MAE
    # tod is normalized to [0, 1), convert to hour bins
    tod_hours = (test_tod * 24).astype(int).clip(0, 23)
    per_tod_mae = np.zeros(24)
    for h in range(24):
        mask = tod_hours == h
        if mask.sum() > 0:
            per_tod_mae[h] = errors[mask].mean()
    result["per_tod_mae"] = per_tod_mae

    # 3. Per-DoW MAE
    dow_days = (test_dow * 7).astype(int).clip(0, 6)
    per_dow_mae = np.zeros(7)
    for d in range(7):
        mask = dow_days == d
        if mask.sum() > 0:
            per_dow_mae[d] = errors[mask].mean()
    result["per_dow_mae"] = per_dow_mae

    # 4. Overall
    result["overall_mae"] = float(errors.mean())
    result["overall_rmse"] = float(np.sqrt(np.mean(errors ** 2)))

    return result


def main():
    years = [2022, 2023, 2024]

    # Load all test data
    test_cache = {}
    for year in years:
        test_data, mean, std = load_data_and_scaler(DATASETS[year])
        test_x, test_y, test_tod, test_dow = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        test_cache[year] = {
            "test_x": test_x, "test_y": test_y,
            "test_tod": test_tod, "test_dow": test_dow,
            "mean": mean, "std": std,
        }
        print(f"Loaded {year} Q1: {len(test_x)} samples, mean={mean:.2f}, std={std:.2f}")

    # Also compute per-node mean flow for context
    node_mean_flow = {}
    for year in years:
        test_y = test_cache[year]["test_y"]  # (N, 12, 893, 1)
        node_mean_flow[year] = test_y.squeeze(-1).mean(axis=(0, 1))  # (893,)

    # Run all pairs: use TRAIN year's scaler (model+scaler are deployed together)
    all_results = {}
    for train_year in years:
        print(f"\n=== Model trained on {train_year} Q1 ===")
        model = load_model(CHECKPOINTS[train_year])
        train_mean = test_cache[train_year]["mean"]
        train_std = test_cache[train_year]["std"]
        for test_year in years:
            c = test_cache[test_year]
            result = analyze_pair(
                train_year, test_year, model,
                c["test_x"], c["test_y"], c["test_tod"], c["test_dow"],
                train_mean, train_std
            )
            key = f"train_{train_year}_test_{test_year}"
            all_results[key] = result
            print(f"  Test on {test_year}: MAE={result['overall_mae']:.4f}")
        del model
        torch.cuda.empty_cache()

    # ==================== ANALYSIS ====================
    print("\n" + "=" * 80)
    print("DETAILED CONCEPT DRIFT ANALYSIS")
    print("=" * 80)

    # Load sensor categories
    dead_idx = np.load("datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy")
    major_fail_idx = np.load("datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy")
    all_problematic = set(dead_idx.tolist()) | set(major_fail_idx.tolist())
    functional_idx = np.array([i for i in range(NUM_NODES) if i not in all_problematic])

    report_lines = []
    def report(s=""):
        print(s)
        report_lines.append(s)

    # ---- Q1: Concentrated vs Gradual Drift ----
    report("\n## Q1: Is drift concentrated in specific nodes or gradual?")
    report()

    # For each cross-year pair, compute per-node degradation vs self-eval
    for test_year in years:
        self_key = f"train_{test_year}_test_{test_year}"
        self_node_mae = all_results[self_key]["per_node_mae"]

        for train_year in years:
            if train_year == test_year:
                continue
            cross_key = f"train_{train_year}_test_{test_year}"
            cross_node_mae = all_results[cross_key]["per_node_mae"]

            # Per-node degradation
            degradation = cross_node_mae - self_node_mae  # (893,)
            deg_func = degradation[functional_idx]

            report(f"### {train_year}→{test_year} (functional nodes only, n={len(functional_idx)})")
            report(f"  Overall MAE: {all_results[self_key]['overall_mae']:.2f} → {all_results[cross_key]['overall_mae']:.2f} (+{all_results[cross_key]['overall_mae'] - all_results[self_key]['overall_mae']:.2f})")
            report(f"  Per-node degradation stats:")
            report(f"    Mean: {deg_func.mean():.4f}")
            report(f"    Median: {np.median(deg_func):.4f}")
            report(f"    Std: {deg_func.std():.4f}")
            report(f"    Max: {deg_func.max():.4f} (node {functional_idx[np.argmax(deg_func)]})")
            report(f"    Min: {deg_func.min():.4f} (node {functional_idx[np.argmin(deg_func)]})")

            # How many nodes improved vs degraded
            n_worse = (deg_func > 0).sum()
            n_better = (deg_func < 0).sum()
            report(f"    Nodes worse: {n_worse}/{len(deg_func)} ({100*n_worse/len(deg_func):.1f}%)")
            report(f"    Nodes better: {n_better}/{len(deg_func)} ({100*n_better/len(deg_func):.1f}%)")

            # Concentration: what % of total degradation comes from worst 10% nodes
            deg_sorted = np.sort(deg_func)[::-1]
            top10pct = int(len(deg_func) * 0.1)
            top10_share = deg_sorted[:top10pct].sum() / max(deg_sorted[deg_sorted > 0].sum(), 1e-10)
            report(f"    Worst 10% nodes account for {100*top10_share:.1f}% of total degradation")

            # Top 10 worst degradation nodes
            worst_idx = functional_idx[np.argsort(degradation[functional_idx])[::-1][:10]]
            report(f"    Top 10 worst nodes: {worst_idx.tolist()}")
            for ni in worst_idx[:5]:
                report(f"      Node {ni}: self={self_node_mae[ni]:.2f} → cross={cross_node_mae[ni]:.2f} (Δ={degradation[ni]:.2f}, mean_flow={node_mean_flow[test_year][ni]:.1f})")
            report()

    # ---- Q2: Worst degradation by ToD and DoW ----
    report("\n## Q2: Degradation by Time-of-Day and Day-of-Week")
    report()

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for test_year in years:
        self_key = f"train_{test_year}_test_{test_year}"
        self_tod = all_results[self_key]["per_tod_mae"]
        self_dow = all_results[self_key]["per_dow_mae"]

        for train_year in years:
            if train_year == test_year:
                continue
            cross_key = f"train_{train_year}_test_{test_year}"
            cross_tod = all_results[cross_key]["per_tod_mae"]
            cross_dow = all_results[cross_key]["per_dow_mae"]

            report(f"### {train_year}→{test_year}")

            # ToD
            tod_deg = cross_tod - self_tod
            worst_hour = np.argmax(tod_deg)
            best_hour = np.argmin(tod_deg)
            report(f"  ToD (worst degradation): {worst_hour}:00 (Δ={tod_deg[worst_hour]:.2f}, self={self_tod[worst_hour]:.2f}→cross={cross_tod[worst_hour]:.2f})")
            report(f"  ToD (least degradation): {best_hour}:00 (Δ={tod_deg[best_hour]:.2f})")

            # Peak hours (7-9, 16-19) vs off-peak
            peak_hours = list(range(7, 10)) + list(range(16, 20))
            off_peak = [h for h in range(24) if h not in peak_hours]
            peak_deg = tod_deg[peak_hours].mean()
            offpeak_deg = tod_deg[off_peak].mean()
            report(f"  Peak hours (7-9,16-19) avg degradation: {peak_deg:.2f}")
            report(f"  Off-peak avg degradation: {offpeak_deg:.2f}")

            # DoW
            dow_deg = cross_dow - self_dow
            worst_day = np.argmax(dow_deg)
            best_day = np.argmin(dow_deg)
            report(f"  DoW (worst): {dow_names[worst_day]} (Δ={dow_deg[worst_day]:.2f})")
            report(f"  DoW (best): {dow_names[best_day]} (Δ={dow_deg[best_day]:.2f})")

            # Weekday vs weekend
            weekday_deg = dow_deg[:5].mean()
            weekend_deg = dow_deg[5:].mean()
            report(f"  Weekday avg degradation: {weekday_deg:.2f}")
            report(f"  Weekend avg degradation: {weekend_deg:.2f}")
            report()

    # ---- Q3: Behavioral Insights ----
    report("\n## Q3: Behavioral Insights")
    report()

    # 3a. Sensor death/revival patterns across years
    report("### 3a. Sensor health changes across years")
    for year in years:
        test_y = test_cache[year]["test_y"].squeeze(-1)  # (N, 12, 893)
        # Per-node zero rate in test set
        zero_rate = (test_y == 0).mean(axis=(0, 1))  # (893,)
        n_dead = (zero_rate > 0.9).sum()
        n_major = ((zero_rate > 0.5) & (zero_rate <= 0.9)).sum()
        n_partial = ((zero_rate > 0.05) & (zero_rate <= 0.5)).sum()
        n_func = (zero_rate <= 0.05).sum()
        report(f"  {year} Q1 test set: dead={n_dead}, major_fail={n_major}, partial={n_partial}, functional={n_func}")

        # Save for cross-comparison
        test_cache[year]["zero_rate"] = zero_rate

    # Sensor state transitions
    report()
    report("### 3b. Sensor state transitions (test set zero_rate)")
    for y1, y2 in [(2022, 2023), (2023, 2024), (2022, 2024)]:
        zr1 = test_cache[y1]["zero_rate"]
        zr2 = test_cache[y2]["zero_rate"]
        # Dead in y1 but alive in y2
        revived = ((zr1 > 0.9) & (zr2 < 0.5)).sum()
        # Alive in y1 but dead in y2
        died = ((zr1 < 0.5) & (zr2 > 0.9)).sum()
        # Both dead
        both_dead = ((zr1 > 0.9) & (zr2 > 0.9)).sum()
        report(f"  {y1}→{y2}: revived={revived}, died={died}, both_dead={both_dead}")

    # 3c. Correlation between node degradation and sensor characteristics
    report()
    report("### 3c. Degradation correlates")
    for train_year, test_year in [(2022, 2023), (2024, 2023), (2022, 2024)]:
        self_key = f"train_{test_year}_test_{test_year}"
        cross_key = f"train_{train_year}_test_{test_year}"
        degradation = all_results[cross_key]["per_node_mae"] - all_results[self_key]["per_node_mae"]

        # Only functional nodes
        deg_f = degradation[functional_idx]
        flow_f = node_mean_flow[test_year][functional_idx]
        zr_f = test_cache[test_year]["zero_rate"][functional_idx]

        # Correlation with mean flow
        corr_flow = np.corrcoef(deg_f, flow_f)[0, 1]
        # Correlation with zero rate
        corr_zr = np.corrcoef(deg_f, zr_f)[0, 1]
        # Correlation with self MAE (do already-hard nodes degrade more?)
        self_mae_f = all_results[self_key]["per_node_mae"][functional_idx]
        corr_self = np.corrcoef(deg_f, self_mae_f)[0, 1]

        report(f"  {train_year}→{test_year} (functional nodes):")
        report(f"    Degradation vs mean_flow: r={corr_flow:.3f}")
        report(f"    Degradation vs zero_rate: r={corr_zr:.3f}")
        report(f"    Degradation vs self_MAE:  r={corr_self:.3f}")

    # 3d. Flow distribution shift
    report()
    report("### 3d. Flow distribution shift across years")
    for year in years:
        test_y = test_cache[year]["test_y"].squeeze(-1)
        func_flow = test_y[:, :, functional_idx]
        report(f"  {year} Q1 (functional): mean={func_flow.mean():.2f}, std={func_flow.std():.2f}, "
               f"median={np.median(func_flow):.2f}, p95={np.percentile(func_flow, 95):.2f}, "
               f"p99={np.percentile(func_flow, 99):.2f}")

    # 3e. Per-node flow change: which nodes changed flow level most?
    report()
    report("### 3e. Nodes with largest flow level change")
    for y1, y2 in [(2022, 2023), (2023, 2024), (2022, 2024)]:
        flow_change = node_mean_flow[y2] - node_mean_flow[y1]
        flow_change_f = flow_change[functional_idx]
        top_increase = functional_idx[np.argsort(flow_change_f)[::-1][:5]]
        top_decrease = functional_idx[np.argsort(flow_change_f)[:5]]
        report(f"  {y1}→{y2}:")
        report(f"    Biggest flow increase:")
        for ni in top_increase:
            report(f"      Node {ni}: {node_mean_flow[y1][ni]:.1f} → {node_mean_flow[y2][ni]:.1f} (Δ={flow_change[ni]:+.1f})")
        report(f"    Biggest flow decrease:")
        for ni in top_decrease:
            report(f"      Node {ni}: {node_mean_flow[y1][ni]:.1f} → {node_mean_flow[y2][ni]:.1f} (Δ={flow_change[ni]:+.1f})")

    # Save all numpy arrays for further analysis
    np.savez(
        os.path.join(OUTPUT_DIR, "drift_analysis_arrays.npz"),
        functional_idx=functional_idx,
        **{f"per_node_mae_{k}": v["per_node_mae"] for k, v in all_results.items()},
        **{f"per_tod_mae_{k}": v["per_tod_mae"] for k, v in all_results.items()},
        **{f"per_dow_mae_{k}": v["per_dow_mae"] for k, v in all_results.items()},
        **{f"node_mean_flow_{y}": node_mean_flow[y] for y in years},
        **{f"zero_rate_{y}": test_cache[y]["zero_rate"] for y in years},
    )
    print(f"\nArrays saved to {OUTPUT_DIR}/drift_analysis_arrays.npz")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "drift_analysis_report.md")
    with open(report_path, "w") as f:
        f.write("# Concept Drift Detailed Analysis (Q1, Unmasked MAE)\n\n")
        f.write("## Cross-Year Evaluation Summary\n\n")
        f.write("| Train \\\\ Test | 2022 Q1 | 2023 Q1 | 2024 Q1 |\n")
        f.write("|---|---|---|---|\n")
        for ty in years:
            row = f"| {ty} Q1 |"
            for tey in years:
                k = f"train_{ty}_test_{tey}"
                mae = all_results[k]["overall_mae"]
                marker = " **" if ty == tey else " "
                end = "**" if ty == tey else ""
                row += f"{marker}{mae:.2f}{end} |"
            f.write(row + "\n")
        f.write("\n")
        for line in report_lines:
            f.write(line + "\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
