"""
Input Signal Analysis (Full Scale): Where Does the Abrupt Change Happen?
=========================================================================
For ALL test incidents across ALL nodes:
  - Compare with matched non-incident at same DOW/TOD
  - Detect WHERE the abrupt change occurs (z-score > threshold)
  - Determine if the change is visible within the model's input window
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = 'eda/post_incident'
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
INCIDENT_PATH = 'datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv'

NUM_NODES = 893
INPUT_LEN = 12
OUTPUT_LEN = 12
DATA_RANGE = (0, 26280)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
STEPS_PER_DAY = 288
RECOVERY_BUFFER = 12
WINDOW = 36  # ±3 hours around incident start
Z_THRESHOLD = 2.0

print("=" * 70)
print("Input Signal Analysis (FULL SCALE): All Test Incidents")
print("=" * 70)

# ============================================================================
# Load data
# ============================================================================
data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, NUM_NODES, 5)[:DATA_RANGE[1]]
flow = data[:, :, 0]
occ = data[:, :, 1]
speed = data[:, :, 2]
tod = data[:, :, 3]
dow = data[:, :, 4]

total_len = DATA_RANGE[1]
test_start = int(total_len * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))

incidents = pd.read_csv(INCIDENT_PATH)
incidents = incidents[incidents['original_start_slot'] < DATA_RANGE[1]].copy()
incidents['incident_end'] = incidents['original_start_slot'] + incidents['duration_slots'] + RECOVERY_BUFFER

# ALL incidents (for matching exclusion)
all_incidents = incidents.copy()

# Test incidents
test_incidents = incidents[
    (incidents['incident_end'] >= test_start) &
    (incidents['original_start_slot'] <= DATA_RANGE[1])
].copy()

print(f"  Total test incidents: {len(test_incidents)}")
print(f"  Nodes with incidents: {test_incidents['sensor_idx'].nunique()}")

# ============================================================================
# Pre-build per-node incident slot sets for fast matching
# ============================================================================
print("\n[0] Building per-node incident slot index...")
node_incident_slots = {}
for node_idx in test_incidents['sensor_idx'].unique():
    node_incs = all_incidents[all_incidents['sensor_idx'] == node_idx]
    slots = set()
    for _, inc in node_incs.iterrows():
        start = int(inc['original_start_slot'])
        end = int(inc['incident_end'])
        for s in range(max(0, start - WINDOW), min(total_len, end + WINDOW)):
            slots.add(s)
    node_incident_slots[node_idx] = slots
print(f"  Indexed {len(node_incident_slots)} nodes")

# ============================================================================
# For each incident, find matched non-incident days and compute change points
# ============================================================================
print("\n[1] Processing all incidents (this may take a few minutes)...")

change_points = []
# For aggregate profile: collect per-timestep deviations
agg_flow_dev = {t: [] for t in range(-WINDOW, WINDOW)}
agg_occ_dev = {t: [] for t in range(-WINDOW, WINDOW)}
agg_speed_dev = {t: [] for t in range(-WINDOW, WINDOW)}
agg_flow_z = {t: [] for t in range(-WINDOW, WINDOW)}
agg_occ_z = {t: [] for t in range(-WINDOW, WINDOW)}
agg_speed_z = {t: [] for t in range(-WINDOW, WINDOW)}

n_processed = 0
n_skipped_no_match = 0

for idx, (_, inc) in enumerate(test_incidents.iterrows()):
    node_idx = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_dur = int(inc['duration_slots'])
    inc_type = inc['incident_type']

    if inc_start >= DATA_RANGE[1]:
        continue

    inc_tod = int(tod[inc_start, node_idx] * STEPS_PER_DAY) % STEPS_PER_DAY
    inc_dow = int(round(dow[inc_start, node_idx] * 7)) % 7

    # Find matched non-incident days at same DOW
    inc_slots = node_incident_slots.get(node_idx, set())
    matched_starts = []
    for day_offset in range(-90, 90):
        if day_offset == 0:
            continue
        candidate_start = inc_start + day_offset * STEPS_PER_DAY
        if candidate_start < WINDOW or candidate_start + WINDOW >= DATA_RANGE[1]:
            continue
        cand_dow = int(round(dow[candidate_start, node_idx] * 7)) % 7
        if cand_dow != inc_dow:
            continue
        # Check no incident near this candidate
        if candidate_start in inc_slots:
            continue
        matched_starts.append(candidate_start)

    if len(matched_starts) < 3:
        n_skipped_no_match += 1
        continue

    # Compute per-timestep stats
    first_sig_t_flow = None
    first_sig_t_occ = None
    first_sig_t_speed = None
    first_sig_t_any = None
    max_delta_t = None
    max_delta_v = 0
    prev_inc_flow = None
    prev_match_flow = None

    for rel_t in range(-WINDOW, WINDOW):
        abs_t = inc_start + rel_t
        if abs_t < 0 or abs_t >= DATA_RANGE[1]:
            continue

        match_flows = []
        match_occs = []
        match_speeds = []
        for ms in matched_starts:
            mt = ms + rel_t
            if 0 <= mt < DATA_RANGE[1]:
                match_flows.append(flow[mt, node_idx])
                match_occs.append(occ[mt, node_idx])
                match_speeds.append(speed[mt, node_idx])

        if len(match_flows) < 3:
            continue

        inc_f = float(flow[abs_t, node_idx])
        inc_o = float(occ[abs_t, node_idx])
        inc_s = float(speed[abs_t, node_idx])
        m_f, m_o, m_s = np.mean(match_flows), np.mean(match_occs), np.mean(match_speeds)
        s_f, s_o, s_s = np.std(match_flows), np.std(match_occs), np.std(match_speeds)

        fz = (inc_f - m_f) / s_f if s_f > 0 else 0
        oz = (inc_o - m_o) / s_o if s_o > 0 else 0
        sz = (inc_s - m_s) / s_s if s_s > 0 else 0

        # Aggregate for profile
        agg_flow_z[rel_t].append(fz)
        agg_occ_z[rel_t].append(oz)
        agg_speed_z[rel_t].append(sz)
        if m_f > 0:
            agg_flow_dev[rel_t].append((inc_f - m_f) / m_f * 100)
        if m_s > 0:
            agg_speed_dev[rel_t].append((inc_s - m_s) / m_s * 100)
        agg_occ_dev[rel_t].append(inc_o - m_o)

        # Change point detection (in [-15, +20] window)
        if -15 <= rel_t <= 20:
            if first_sig_t_flow is None and abs(fz) > Z_THRESHOLD:
                first_sig_t_flow = rel_t
            if first_sig_t_occ is None and abs(oz) > Z_THRESHOLD:
                first_sig_t_occ = rel_t
            if first_sig_t_speed is None and abs(sz) > Z_THRESHOLD:
                first_sig_t_speed = rel_t
            if first_sig_t_any is None and (abs(fz) > Z_THRESHOLD or abs(oz) > Z_THRESHOLD or abs(sz) > Z_THRESHOLD):
                first_sig_t_any = rel_t

            # Max consecutive delta
            if prev_inc_flow is not None and prev_match_flow is not None:
                delta_diff = abs((inc_f - prev_inc_flow) - (m_f - prev_match_flow))
                if delta_diff > max_delta_v:
                    max_delta_v = delta_diff
                    max_delta_t = rel_t

        prev_inc_flow = inc_f
        prev_match_flow = m_f

    change_points.append({
        'node_idx': node_idx,
        'incident_type': inc_type,
        'incident_duration': inc_dur,
        'first_sig_flow': first_sig_t_flow,
        'first_sig_occ': first_sig_t_occ,
        'first_sig_speed': first_sig_t_speed,
        'first_sig_any': first_sig_t_any,
        'max_delta_t': max_delta_t,
        'max_delta_value': max_delta_v,
        'sig_flow_in_input': first_sig_t_flow is not None and -INPUT_LEN <= first_sig_t_flow < 0,
        'sig_occ_in_input': first_sig_t_occ is not None and -INPUT_LEN <= first_sig_t_occ < 0,
        'sig_speed_in_input': first_sig_t_speed is not None and -INPUT_LEN <= first_sig_t_speed < 0,
        'sig_any_in_input': first_sig_t_any is not None and -INPUT_LEN <= first_sig_t_any < 0,
    })

    n_processed += 1
    if (n_processed % 200) == 0:
        print(f"    Processed {n_processed} incidents...")

print(f"\n  Processed: {n_processed}, Skipped (no match): {n_skipped_no_match}")

# ============================================================================
# Aggregate analysis
# ============================================================================
print("\n[2] Aggregate change point statistics...")

df_cp = pd.DataFrame(change_points)
N = len(df_cp)

print(f"\n  Total incidents analyzed: {N}")
print(f"\n  === Signal Detection in Input Window [-12, 0) ===")
for var in ['flow', 'occ', 'speed', 'any']:
    col = f'sig_{var}_in_input'
    n_sig = df_cp[col].sum()
    print(f"    {var:>6}: {n_sig:4d}/{N} ({n_sig/N*100:5.1f}%) have |z|>{Z_THRESHOLD} in input window")

print(f"\n  === Signal Detection Anywhere [-15, +20] ===")
for var, col in [('flow', 'first_sig_flow'), ('occ', 'first_sig_occ'),
                  ('speed', 'first_sig_speed'), ('any', 'first_sig_any')]:
    has_sig = df_cp[col].notna().sum()
    print(f"    {var:>6}: {has_sig:4d}/{N} ({has_sig/N*100:5.1f}%) have |z|>{Z_THRESHOLD} somewhere")

# Distribution of first significant change
print(f"\n  === Distribution of First Significant Change (any variable) ===")
has_sig = df_cp[df_cp['first_sig_any'].notna()].copy()
if len(has_sig) > 0:
    bins = [-16, -12, -6, 0, 6, 12, 21]
    labels = ['before_input(-15,-12)', 'early_input(-12,-6)', 'late_input(-6,0)',
              'early_pred(0,6)', 'mid_pred(6,12)', 'late_pred(12,20)']
    has_sig['time_bin'] = pd.cut(has_sig['first_sig_any'], bins=bins, labels=labels, right=False)
    for label in labels:
        cnt = (has_sig['time_bin'] == label).sum()
        print(f"    {label:>25}: {cnt:4d} ({cnt/N*100:5.1f}% of all)")

# Breakdown by incident duration
print(f"\n  === Signal Rate by Incident Duration ===")
dur_bins = [(0, 2, 'short(0-10min)'), (2, 6, 'medium(10-30min)'), (6, 20, 'long(30-100min)'), (20, 999, 'very_long(100min+)')]
for lo, hi, label in dur_bins:
    mask = (df_cp['incident_duration'] >= lo) & (df_cp['incident_duration'] < hi)
    sub = df_cp[mask]
    if len(sub) > 0:
        sig_rate = sub['sig_any_in_input'].mean() * 100
        print(f"    {label:>25}: {len(sub):4d} incidents, {sig_rate:5.1f}% have signal in input")

# Breakdown by incident type
print(f"\n  === Signal Rate by Incident Type (top 10) ===")
type_stats = df_cp.groupby('incident_type').agg(
    count=('node_idx', 'count'),
    sig_any_pct=('sig_any_in_input', 'mean'),
    sig_flow_pct=('sig_flow_in_input', 'mean'),
    sig_speed_pct=('sig_speed_in_input', 'mean'),
).sort_values('count', ascending=False)
for itype, row in type_stats.head(10).iterrows():
    print(f"    {itype:>15}: n={int(row['count']):4d}, any={row['sig_any_pct']*100:5.1f}%, "
          f"flow={row['sig_flow_pct']*100:5.1f}%, speed={row['sig_speed_pct']*100:5.1f}%")

# ============================================================================
# Build aggregate temporal profile
# ============================================================================
print("\n[3] Building aggregate temporal profiles...")

profile_rows = []
for t in range(-WINDOW, WINDOW):
    row = {'relative_time': t}
    for name, agg in [('flow_z', agg_flow_z), ('occ_z', agg_occ_z), ('speed_z', agg_speed_z),
                       ('flow_dev_pct', agg_flow_dev), ('occ_dev', agg_occ_dev), ('speed_dev_pct', agg_speed_dev)]:
        vals = agg[t]
        if len(vals) > 0:
            row[f'{name}_mean'] = np.mean(vals)
            row[f'{name}_median'] = np.median(vals)
            row[f'{name}_q25'] = np.percentile(vals, 25)
            row[f'{name}_q75'] = np.percentile(vals, 75)
            row[f'{name}_std'] = np.std(vals)
            row['count'] = len(vals)
        else:
            row[f'{name}_mean'] = np.nan
            row[f'{name}_median'] = np.nan
    profile_rows.append(row)
profile = pd.DataFrame(profile_rows)

print("\n  Timestep-level deviation (mean z-score) around incident start:")
print(f"  {'t':>4} | {'Flow Z':>8} {'Occ Z':>8} {'Spd Z':>8} | {'Flow%':>8} {'Spd%':>8}")
print("  " + "-" * 60)
for _, r in profile[(profile['relative_time'] >= -14) & (profile['relative_time'] <= 14)].iterrows():
    t = int(r['relative_time'])
    marker = ''
    if t == -12: marker = ' <-- INPUT START'
    elif t == 0: marker = ' <-- INC START'
    print(f"  {t:4d} | {r.get('flow_z_mean',0):8.3f} {r.get('occ_z_mean',0):8.3f} {r.get('speed_z_mean',0):8.3f} "
          f"| {r.get('flow_dev_pct_mean',0):7.2f}% {r.get('speed_dev_pct_mean',0):7.2f}%{marker}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[4] Creating visualizations...")

fig = plt.figure(figsize=(28, 36))
gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle(f'Input Signal Analysis: ALL {N} Test Incidents ({n_skipped_no_match} skipped, no match)',
             fontsize=16, fontweight='bold', y=0.98)

# --- Row 1: Mean z-score profiles ---
for i, (var, label) in enumerate([('flow_z', 'Flow'), ('occ_z', 'Occupancy'), ('speed_z', 'Speed')]):
    ax = fig.add_subplot(gs[0, i])
    t = profile['relative_time']
    mean_col = f'{var}_mean'
    q25_col = f'{var}_q25'
    q75_col = f'{var}_q75'
    ax.plot(t, profile[mean_col], 'darkblue', linewidth=2, label='Mean Z-score')
    ax.plot(t, profile.get(f'{var}_median', profile[mean_col]), 'blue', linewidth=1, alpha=0.5, label='Median Z-score')
    if q25_col in profile.columns:
        ax.fill_between(t, profile[q25_col], profile[q75_col], alpha=0.15, color='blue', label='IQR (25-75%)')
    ax.axhline(y=Z_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'z=±{Z_THRESHOLD}')
    ax.axhline(y=-Z_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvspan(-INPUT_LEN, 0, alpha=0.08, color='blue')
    ax.axvspan(0, OUTPUT_LEN, alpha=0.08, color='orange')
    ax.set_xlabel('Relative Time (5-min slots)')
    ax.set_ylabel('Z-score')
    ax.set_title(f'{label} Z-score (N={N})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-WINDOW, WINDOW)

# --- Row 2: Deviation % profiles ---
for i, (var, label, unit) in enumerate([('flow_dev_pct', 'Flow', '%'),
                                         ('occ_dev', 'Occupancy', 'raw'),
                                         ('speed_dev_pct', 'Speed', '%')]):
    ax = fig.add_subplot(gs[1, i])
    mean_col = f'{var}_mean'
    q25_col = f'{var}_q25'
    q75_col = f'{var}_q75'
    ax.plot(profile['relative_time'], profile[mean_col], 'purple', linewidth=2, label='Mean')
    if q25_col in profile.columns:
        ax.fill_between(profile['relative_time'], profile[q25_col], profile[q75_col],
                        alpha=0.15, color='purple', label='IQR')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvspan(-INPUT_LEN, 0, alpha=0.08, color='blue')
    ax.axvspan(0, OUTPUT_LEN, alpha=0.08, color='orange')
    ax.set_xlabel('Relative Time')
    ax.set_ylabel(f'Deviation ({unit})')
    ax.set_title(f'{label} Deviation from Matched Normal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-WINDOW, WINDOW)

# --- Row 3: Change point histograms ---
for i, (var, label) in enumerate([('first_sig_flow', 'Flow'), ('first_sig_occ', 'Occupancy'), ('first_sig_speed', 'Speed')]):
    ax = fig.add_subplot(gs[2, i])
    sig_vals = df_cp[var].dropna()
    if len(sig_vals) > 0:
        ax.hist(sig_vals, bins=range(-15, 21), color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=-12, color='blue', linestyle='--', linewidth=2, label='Input start')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Incident start')
    n_in = df_cp[f'sig_{var.replace("first_sig_","")}_in_input'].sum()
    ax.set_xlabel('Relative Time of First |z|>2')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}: First Significant Change\n(in input: {n_in}/{N} = {n_in/N*100:.1f}%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# --- Row 4: Combined any-variable histogram + duration breakdown + summary ---
ax = fig.add_subplot(gs[3, 0])
sig_any = df_cp['first_sig_any'].dropna()
if len(sig_any) > 0:
    ax.hist(sig_any, bins=range(-15, 21), color='darkorange', alpha=0.7, edgecolor='black')
ax.axvline(x=-12, color='blue', linestyle='--', linewidth=2, label='Input start')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Incident start')
ax.set_xlabel('Relative Time')
ax.set_ylabel('Count')
n_any_input = df_cp['sig_any_in_input'].sum()
ax.set_title(f'Any Variable: First |z|>2\n(in input: {n_any_input}/{N} = {n_any_input/N*100:.1f}%)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Duration breakdown
ax = fig.add_subplot(gs[3, 1])
dur_labels, dur_rates, dur_counts = [], [], []
for lo, hi, label in dur_bins:
    mask = (df_cp['incident_duration'] >= lo) & (df_cp['incident_duration'] < hi)
    sub = df_cp[mask]
    if len(sub) > 0:
        dur_labels.append(label)
        dur_rates.append(sub['sig_any_in_input'].mean() * 100)
        dur_counts.append(len(sub))
bars = ax.bar(range(len(dur_labels)), dur_rates, color=['#4CAF50', '#2196F3', '#FF9800', '#f44336'], alpha=0.8)
ax.set_xticks(range(len(dur_labels)))
ax.set_xticklabels([l.split('(')[0] for l in dur_labels], fontsize=9)
for bar, rate, cnt in zip(bars, dur_rates, dur_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{rate:.1f}%\n(n={cnt})', ha='center', fontsize=8)
ax.set_ylabel('% with Signal in Input Window')
ax.set_title('Signal Rate by Incident Duration')
ax.grid(True, alpha=0.3, axis='y')

# Summary text
ax = fig.add_subplot(gs[3, 2])
ax.axis('off')
summary_lines = [
    f"FULL-SCALE SIGNAL DETECTION SUMMARY",
    f"=" * 45,
    f"Total incidents analyzed: {N}",
    f"Skipped (no match): {n_skipped_no_match}",
    f"",
    f"SIGNAL IN INPUT WINDOW [-12, 0):",
    f"  Flow only:     {df_cp['sig_flow_in_input'].sum():4d}/{N} ({df_cp['sig_flow_in_input'].mean()*100:.1f}%)",
    f"  Occ only:      {df_cp['sig_occ_in_input'].sum():4d}/{N} ({df_cp['sig_occ_in_input'].mean()*100:.1f}%)",
    f"  Speed only:    {df_cp['sig_speed_in_input'].sum():4d}/{N} ({df_cp['sig_speed_in_input'].mean()*100:.1f}%)",
    f"  Any variable:  {df_cp['sig_any_in_input'].sum():4d}/{N} ({df_cp['sig_any_in_input'].mean()*100:.1f}%)",
    f"",
    f"SIGNAL ANYWHERE [-15, +20]:",
    f"  Flow:   {df_cp['first_sig_flow'].notna().sum():4d}/{N} ({df_cp['first_sig_flow'].notna().mean()*100:.1f}%)",
    f"  Occ:    {df_cp['first_sig_occ'].notna().sum():4d}/{N} ({df_cp['first_sig_occ'].notna().mean()*100:.1f}%)",
    f"  Speed:  {df_cp['first_sig_speed'].notna().sum():4d}/{N} ({df_cp['first_sig_speed'].notna().mean()*100:.1f}%)",
    f"  Any:    {df_cp['first_sig_any'].notna().sum():4d}/{N} ({df_cp['first_sig_any'].notna().mean()*100:.1f}%)",
]
ax.text(0.02, 0.98, '\n'.join(summary_lines), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Row 5: Type breakdown ---
ax = fig.add_subplot(gs[4, 0:2])
top_types = type_stats.head(8)
x = range(len(top_types))
w = 0.25
ax.bar([xi - w for xi in x], top_types['sig_any_pct'] * 100, w, label='Any var', color='darkorange', alpha=0.8)
ax.bar([xi for xi in x], top_types['sig_flow_pct'] * 100, w, label='Flow', color='blue', alpha=0.8)
ax.bar([xi + w for xi in x], top_types['sig_speed_pct'] * 100, w, label='Speed', color='green', alpha=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([f'{t}\n(n={int(top_types.loc[t,"count"])})' for t in top_types.index], fontsize=8)
ax.set_ylabel('% with Signal in Input')
ax.set_title('Signal Detection Rate by Incident Type')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Fraction of incidents with signal that are before vs after t=0
ax = fig.add_subplot(gs[4, 2])
for var, color, label in [('first_sig_flow', 'blue', 'Flow'),
                            ('first_sig_occ', 'orange', 'Occ'),
                            ('first_sig_speed', 'green', 'Speed')]:
    vals = df_cp[var].dropna()
    if len(vals) > 0:
        before = (vals < 0).sum()
        after = (vals >= 0).sum()
        ax.barh(label, before, color=color, alpha=0.6, label='Before inc (input)')
        ax.barh(label, -after, color=color, alpha=0.3)
        ax.text(before + 2, label, f'{before}', va='center', fontsize=9)
        ax.text(-after - 2, label, f'{after}', va='center', ha='right', fontsize=9)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('← After incident | Before incident →')
ax.set_title('Signal Timing: Before vs After Incident Start')
ax.grid(True, alpha=0.3, axis='x')

# --- Row 6-7: Sample case studies (12 random incidents with signal) ---
sig_incidents = df_cp[df_cp['sig_any_in_input']].copy()
if len(sig_incidents) > 12:
    sample_cp = sig_incidents.sample(12, random_state=42)
else:
    sample_cp = sig_incidents

# For case studies, we need to recompute per-timestep data for these specific incidents
case_idx = 0
for row_offset in range(2):
    for col_idx in range(3):
        plot_idx = row_offset * 3 + col_idx
        if plot_idx >= len(sample_cp):
            break
        cp_row = sample_cp.iloc[plot_idx]
        nid = int(cp_row['node_idx'])
        itype = cp_row['incident_type']
        idur = int(cp_row['incident_duration'])

        # Find this incident in test_incidents
        match_incs = test_incidents[
            (test_incidents['sensor_idx'] == nid) &
            (test_incidents['incident_type'] == itype) &
            (test_incidents['duration_slots'] == idur)
        ]
        if len(match_incs) == 0:
            continue
        inc = match_incs.iloc[0]
        inc_start = int(inc['original_start_slot'])

        # Get matched days
        inc_dow_val = int(round(dow[inc_start, nid] * 7)) % 7
        inc_slots = node_incident_slots.get(nid, set())
        matched_starts = []
        for day_offset in range(-90, 90):
            if day_offset == 0:
                continue
            cs = inc_start + day_offset * STEPS_PER_DAY
            if cs < WINDOW or cs + WINDOW >= DATA_RANGE[1]:
                continue
            cd = int(round(dow[cs, nid] * 7)) % 7
            if cd != inc_dow_val:
                continue
            if cs in inc_slots:
                continue
            matched_starts.append(cs)
        if len(matched_starts) < 3:
            continue

        # Collect timestep data
        ts, inc_flows, match_means, match_stds = [], [], [], []
        for rel_t in range(-WINDOW, WINDOW):
            abs_t = inc_start + rel_t
            if abs_t < 0 or abs_t >= DATA_RANGE[1]:
                continue
            mf = [flow[ms + rel_t, nid] for ms in matched_starts if 0 <= ms + rel_t < DATA_RANGE[1]]
            if len(mf) < 3:
                continue
            ts.append(rel_t)
            inc_flows.append(float(flow[abs_t, nid]))
            match_means.append(np.mean(mf))
            match_stds.append(np.std(mf))

        ax = fig.add_subplot(gs[5 + row_offset, col_idx])
        ax.plot(ts, inc_flows, 'r-', linewidth=2, label='Incident')
        ax.plot(ts, match_means, 'g-', linewidth=1.5, label='Matched')
        ax.fill_between(ts, np.array(match_means) - np.array(match_stds),
                        np.array(match_means) + np.array(match_stds),
                        alpha=0.15, color='green')
        ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvspan(-INPUT_LEN, 0, alpha=0.08, color='blue')
        ax.axvspan(0, min(idur + RECOVERY_BUFFER, WINDOW), alpha=0.1, color='red')

        sig_t = cp_row['first_sig_any']
        if pd.notna(sig_t):
            ax.axvline(x=sig_t, color='purple', linestyle='--', linewidth=1.5,
                      label=f'First |z|>2 @ t={int(sig_t)}')

        ax.set_xlabel('Relative Time (5-min)')
        ax.set_title(f'Node {nid}: {itype}, dur={idur*5}min\n'
                     f'Signal@t={int(sig_t) if pd.notna(sig_t) else "none"}', fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

plt.savefig(f'{OUTPUT_DIR}/input_signal_analysis_full.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/input_signal_analysis_full.png")

# Save data
profile.to_csv(f'{OUTPUT_DIR}/timestep_deviation_profile_full.csv', index=False)
df_cp.to_csv(f'{OUTPUT_DIR}/change_point_detection_full.csv', index=False)
print(f"Saved: timestep_deviation_profile_full.csv, change_point_detection_full.csv")

print("\n" + "=" * 70)
print("FULL-SCALE INPUT SIGNAL ANALYSIS COMPLETE")
print("=" * 70)
