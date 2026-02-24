"""Analyze concept drift on FUNCTIONAL nodes (dead/fault excluded).

Criterion: zero_rate < 20% in all 3 years.
- zero_rate < 5% is too strict (nighttime zero on low-traffic roads is legitimate)
- Flow change filter removed (it removes the very drift we want to measure)
- 20% threshold: excludes dead (>90%) and major fault (50-90%), keeps functional sensors
"""
import numpy as np

data = np.load("eda/concept_drift/drift_analysis_arrays.npz")
zr_2022 = data["zero_rate_2022"]
zr_2023 = data["zero_rate_2023"]
zr_2024 = data["zero_rate_2024"]

years = [2022, 2023, 2024]
mf = {y: data[f"node_mean_flow_{y}"] for y in years}

# ========== Functional nodes: zero_rate < 20% in ALL years ==========
func_mask = (zr_2022 < 0.20) & (zr_2023 < 0.20) & (zr_2024 < 0.20)
func_idx = np.where(func_mask)[0]
print(f"=== Functional nodes (zero_rate < 20% all years): {len(func_idx)} / 893 ===")

# Also show per-year counts for context
for y, zr in [(2022, zr_2022), (2023, zr_2023), (2024, zr_2024)]:
    n5 = (zr < 0.05).sum()
    n10 = (zr < 0.10).sum()
    n20 = (zr < 0.20).sum()
    n50 = (zr < 0.50).sum()
    print(f"  {y}: <5%={n5}, <10%={n10}, <20%={n20}, <50%={n50}, total=893")

# ========== Cross-year MAE matrix ==========
def print_matrix(label, idx):
    print(f"\n{'='*60}")
    print(f"{label} (n={len(idx)})")
    print(f"{'='*60}")
    header = "Train / Test"
    print(f"{header:<15} {'2022':>10} {'2023':>10} {'2024':>10}")
    print("-" * 45)
    for ty in years:
        row = f"{ty} Q1       "
        for tey in years:
            k = f"per_node_mae_train_{ty}_test_{tey}"
            mae = data[k][idx].mean()
            marker = "*" if ty == tey else " "
            row += f"{mae:>9.2f}{marker}"
        print(row)

    # Degradation summary
    print(f"\n  Degradation (Cross - Self):")
    pairs = [(2022, 2023), (2022, 2024), (2023, 2022), (2023, 2024), (2024, 2022), (2024, 2023)]
    print(f"  {'Pair':<12} {'Self':>8} {'Cross':>8} {'Delta':>8} {'Delta%':>8}")
    print(f"  {'-'*46}")
    for train_y, test_y in pairs:
        self_k = f"per_node_mae_train_{test_y}_test_{test_y}"
        cross_k = f"per_node_mae_train_{train_y}_test_{test_y}"
        s = data[self_k][idx].mean()
        c = data[cross_k][idx].mean()
        d = c - s
        pct = 100 * d / s
        print(f"  {train_y}->{test_y:<7} {s:>8.2f} {c:>8.2f} {d:>+8.2f} {pct:>+7.1f}%")

print_matrix("ALL nodes", np.arange(893))
print_matrix("Functional nodes (zero_rate < 20%)", func_idx)

# ========== nMAE analysis ==========
print(f"\n{'='*60}")
print("nMAE ANALYSIS (MAE / mean_flow): scale-normalized")
print(f"{'='*60}")

for label, idx in [("All 893", np.arange(893)), ("Functional", func_idx)]:
    print(f"\n--- {label} (n={len(idx)}) ---")
    pairs = [(2022, 2023), (2022, 2024), (2023, 2022), (2023, 2024), (2024, 2022), (2024, 2023)]
    print(f"  {'Pair':<12} {'Self nMAE':>10} {'Cross nMAE':>11} {'Delta':>8} {'Delta%':>8}")
    print(f"  {'-'*52}")
    for train_y, test_y in pairs:
        self_k = f"per_node_mae_train_{test_y}_test_{test_y}"
        cross_k = f"per_node_mae_train_{train_y}_test_{test_y}"
        node_flow = mf[test_y][idx]
        # Exclude near-zero flow nodes for nMAE
        valid = node_flow > 5
        if valid.sum() < 10:
            continue
        s_nmae = (data[self_k][idx][valid] / node_flow[valid]).mean()
        c_nmae = (data[cross_k][idx][valid] / node_flow[valid]).mean()
        d = c_nmae - s_nmae
        pct = 100 * d / s_nmae
        print(f"  {train_y}->{test_y:<7} {s_nmae:>10.4f} {c_nmae:>11.4f} {d:>+8.4f} {pct:>+7.1f}%")

# ========== Per-node degradation distribution ==========
print(f"\n{'='*60}")
print(f"Per-node degradation distribution (Functional, n={len(func_idx)})")
print(f"{'='*60}")
pairs = [(2022, 2023), (2022, 2024), (2023, 2022), (2023, 2024), (2024, 2022), (2024, 2023)]
for train_y, test_y in pairs:
    self_k = f"per_node_mae_train_{test_y}_test_{test_y}"
    cross_k = f"per_node_mae_train_{train_y}_test_{test_y}"
    deg = data[cross_k][func_idx] - data[self_k][func_idx]
    n_worse = (deg > 0).sum()
    print(f"  {train_y}->{test_y}: worse={n_worse}/{len(func_idx)} ({100*n_worse/len(func_idx):.0f}%), "
          f"mean={deg.mean():.2f}, median={np.median(deg):.2f}, max={deg.max():.2f}")

# ========== Flow stats ==========
print(f"\n{'='*60}")
print(f"Flow stats on functional nodes (n={len(func_idx)})")
print(f"{'='*60}")
for year in years:
    f = mf[year][func_idx]
    print(f"  {year}: mean={f.mean():.1f}, std={f.std():.1f}, median={np.median(f):.1f}, "
          f"min={f.min():.1f}, max={f.max():.1f}")

# ========== Zero rate distribution of functional nodes ==========
print(f"\n{'='*60}")
print(f"Zero rate distribution of functional nodes")
print(f"{'='*60}")
for year, zr in [(2022, zr_2022), (2023, zr_2023), (2024, zr_2024)]:
    zr_f = zr[func_idx]
    print(f"  {year}: mean={zr_f.mean():.3f}, median={np.median(zr_f):.3f}, "
          f"max={zr_f.max():.3f}, >10%={( zr_f > 0.10).sum()}")

# Save functional indices
np.save("eda/concept_drift/functional_indices.npy", func_idx)
print(f"\nFunctional indices saved to eda/concept_drift/functional_indices.npy")
