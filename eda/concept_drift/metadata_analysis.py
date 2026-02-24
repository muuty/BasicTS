"""Analyze sensor metadata and incidents to explain 2023 anomaly.

Key question: Why is 2023 Q1 so different from 2022/2024 Q1?
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========== 1. Load sensor metadata ==========
meta = pd.read_csv('/data/XTraffic/process/data/counties/SAN_BERNARDINO/metadata.csv')
print(f"=== Sensor Metadata: {len(meta)} sensors ===")
print(f"Freeways: {meta['Fwy'].unique()}")
print(f"Types: {meta['Type'].value_counts().to_dict()}")
print(f"Cities: {meta['City'].nunique()} unique cities")

# ========== 2. Load drift data ==========
drift = np.load("eda/concept_drift/drift_analysis_arrays.npz")
func_idx = np.load("eda/concept_drift/functional_indices.npy")
years = [2022, 2023, 2024]
mf = {y: drift[f"node_mean_flow_{y}"] for y in years}
zr = {y: drift[f"zero_rate_{y}"] for y in years}

# ========== 3. Load and filter incidents to Q1 (Jan-Mar) ==========
print(f"\n{'='*60}")
print("INCIDENT ANALYSIS: Q1 (Jan-Mar) comparison")
print(f"{'='*60}")

q1_incidents = {}
for y in years:
    df = pd.read_csv(f'/data/XTraffic/process/data/incidents_y{y}.csv', sep='\t')
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    # Filter Q1: Jan-Mar
    q1 = df[(df['dt'].dt.month >= 1) & (df['dt'].dt.month <= 3)]
    # Filter San Bernardino area (check AREA or use Fwy matching)
    # The Fwy numbers in metadata: 10, 15, 60, 71, 210
    sb_fwys = meta['Fwy'].unique()
    q1_sb = q1[q1['Fwy'].isin(sb_fwys)]
    q1_incidents[y] = q1_sb

    print(f"\n  {y} Q1:")
    print(f"    Total incidents (all counties): {len(q1)}")
    print(f"    SB-area freeways: {len(q1_sb)}")
    print(f"    By type: {q1_sb['Type'].value_counts().to_dict()}")
    print(f"    By freeway: {q1_sb['Fwy'].value_counts().to_dict()}")
    print(f"    Date range: {q1_sb['dt'].min()} ~ {q1_sb['dt'].max()}")

# ========== 4. Per-freeway drift analysis ==========
print(f"\n{'='*60}")
print("PER-FREEWAY DRIFT ANALYSIS")
print(f"{'='*60}")

for fwy in sorted(meta['Fwy'].unique()):
    fwy_idx = meta[meta['Fwy'] == fwy]['idx'].values
    # Intersect with functional
    fwy_func = np.intersect1d(fwy_idx, func_idx)
    if len(fwy_func) < 5:
        continue

    print(f"\n--- Fwy {fwy} (total={len(fwy_idx)}, functional={len(fwy_func)}) ---")

    # Mean flow per year
    for y in years:
        f = mf[y][fwy_func].mean()
        z = zr[y][fwy_func].mean()
        print(f"  {y}: mean_flow={f:.1f}, mean_zero_rate={z:.3f}")

    # Cross-year MAE
    print(f"  {'Pair':<12} {'Self':>8} {'Cross':>8} {'Delta%':>8}")
    for ty, tey in [(2022,2023), (2022,2024), (2023,2024)]:
        self_k1 = f"per_node_mae_train_{ty}_test_{ty}"
        self_k2 = f"per_node_mae_train_{tey}_test_{tey}"
        cross_k = f"per_node_mae_train_{ty}_test_{tey}"
        cross_k2 = f"per_node_mae_train_{tey}_test_{ty}"

        s1 = drift[self_k1][fwy_func].mean()
        c12 = drift[cross_k][fwy_func].mean()
        s2 = drift[self_k2][fwy_func].mean()
        c21 = drift[cross_k2][fwy_func].mean()

        pct_12 = 100 * (c12 - s2) / s2
        pct_21 = 100 * (c21 - s1) / s1
        print(f"  {ty}->{tey:<7} {s2:>8.2f} {c12:>8.2f} {pct_12:>+7.1f}%")
        print(f"  {tey}->{ty:<7} {s1:>8.2f} {c21:>8.2f} {pct_21:>+7.1f}%")

# ========== 5. Per-sensor-type drift analysis ==========
print(f"\n{'='*60}")
print("PER-SENSOR-TYPE DRIFT ANALYSIS")
print(f"{'='*60}")

for stype in meta['Type'].unique():
    type_idx = meta[meta['Type'] == stype]['idx'].values
    type_func = np.intersect1d(type_idx, func_idx)
    if len(type_func) < 5:
        continue

    print(f"\n--- {stype} (total={len(type_idx)}, functional={len(type_func)}) ---")
    for ty, tey in [(2022,2023), (2022,2024)]:
        self_k = f"per_node_mae_train_{tey}_test_{tey}"
        cross_k = f"per_node_mae_train_{ty}_test_{tey}"
        s = drift[self_k][type_func].mean()
        c = drift[cross_k][type_func].mean()
        pct = 100 * (c - s) / s
        print(f"  {ty}->{tey}: self={s:.2f}, cross={c:.2f}, delta={pct:+.1f}%")

# ========== 6. Flow change 2022->2023 vs 2022->2024 by freeway ==========
print(f"\n{'='*60}")
print("FLOW CHANGE BY FREEWAY (functional nodes only)")
print(f"{'='*60}")

print(f"{'Fwy':>5} {'n':>4} {'flow22':>8} {'flow23':>8} {'flow24':>8} {'chg22-23':>9} {'chg22-24':>9}")
print("-" * 55)
for fwy in sorted(meta['Fwy'].unique()):
    fwy_func = np.intersect1d(meta[meta['Fwy'] == fwy]['idx'].values, func_idx)
    if len(fwy_func) < 3:
        continue
    f22 = mf[2022][fwy_func].mean()
    f23 = mf[2023][fwy_func].mean()
    f24 = mf[2024][fwy_func].mean()
    chg23 = 100 * (f23 - f22) / f22
    chg24 = 100 * (f24 - f22) / f22
    print(f"{fwy:>5} {len(fwy_func):>4} {f22:>8.1f} {f23:>8.1f} {f24:>8.1f} {chg23:>+8.1f}% {chg24:>+8.1f}%")

# ========== 7. Worst drift nodes: what are they? ==========
print(f"\n{'='*60}")
print("TOP 20 WORST DRIFT NODES (2022->2023, functional only)")
print(f"{'='*60}")

self_k = "per_node_mae_train_2023_test_2023"
cross_k = "per_node_mae_train_2022_test_2023"
deg = drift[cross_k][func_idx] - drift[self_k][func_idx]
worst_order = np.argsort(deg)[::-1][:20]

print(f"{'rank':>4} {'idx':>5} {'Fwy':>5} {'Type':>10} {'City':>15} {'Name':>20} "
      f"{'self':>6} {'cross':>7} {'delta':>7} {'zr23':>6} {'flow23':>8}")
print("-" * 105)
for rank, wi in enumerate(worst_order):
    node = func_idx[wi]
    row = meta[meta['idx'] == node].iloc[0]
    s = drift[self_k][node]
    c = drift[cross_k][node]
    d = c - s
    print(f"{rank+1:>4} {node:>5} {row['Fwy']:>5} {row['Type']:>10} {str(row['City'])[:15]:>15} "
          f"{str(row['Name'])[:20]:>20} {s:>6.1f} {c:>7.1f} {d:>+7.1f} {zr[2023][node]:>.3f} {mf[2023][node]:>8.1f}")

# ========== 8. Zero rate changes: which nodes became problematic in 2023? ==========
print(f"\n{'='*60}")
print("SENSORS THAT BECAME PROBLEMATIC IN 2023 (zero_rate spike)")
print(f"{'='*60}")

# Nodes where 2023 zero_rate > 2x of max(2022, 2024) zero_rate
zr_spike = (zr[2023] > 2 * np.maximum(zr[2022], zr[2024])) & (zr[2023] > 0.05)
spike_idx = np.where(zr_spike)[0]
print(f"Nodes with 2023 zero_rate spike (>2x of max(2022,2024) and >5%): {len(spike_idx)}")

if len(spike_idx) > 0:
    print(f"{'idx':>5} {'Fwy':>5} {'Type':>10} {'City':>15} {'zr22':>6} {'zr23':>6} {'zr24':>6} {'flow22':>8} {'flow23':>8} {'flow24':>8}")
    print("-" * 95)
    for node in spike_idx[:30]:
        row = meta[meta['idx'] == node].iloc[0]
        print(f"{node:>5} {row['Fwy']:>5} {row['Type']:>10} {str(row['City'])[:15]:>15} "
              f"{zr[2022][node]:>6.3f} {zr[2023][node]:>6.3f} {zr[2024][node]:>6.3f} "
              f"{mf[2022][node]:>8.1f} {mf[2023][node]:>8.1f} {mf[2024][node]:>8.1f}")

# Reverse: nodes problematic in 2022/2024 but fine in 2023
zr_dip = (zr[2023] < 0.5 * np.minimum(zr[2022], zr[2024])) & (np.minimum(zr[2022], zr[2024]) > 0.05)
dip_idx = np.where(zr_dip)[0]
print(f"\nNodes RECOVERED in 2023 (zero_rate < 0.5x of min(2022,2024) where others >5%): {len(dip_idx)}")
