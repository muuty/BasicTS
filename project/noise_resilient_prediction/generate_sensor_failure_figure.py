import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Create output directory
output_dir = '/data/pretrainingbasicts/project/noise_resilient_prediction/figures'
os.makedirs(output_dir, exist_ok=True)

print("[OBJECTIVE] Create presentation-quality figure showing sensor failure types")

# Load data
print("[DATA] Loading SAN_BERNARDINO data...")
data = np.memmap('/data/pretrainingbasicts/datasets/xtraffic/SAN_BERNARDINO/data.dat',
                 dtype='float32', mode='r').reshape(105120, 893, 5)

# Use Node 424 as functional sensor, 3 days starting from day 5
start = 288 * 5
end = start + 288 * 3
functional_flow = data[start:end, 424, 0].copy()

# Dead sensor: Node 2
dead_flow = data[start:end, 2, 0].copy()

# Time axis in hours
hours = np.arange(len(functional_flow)) * 5 / 60

print(f"[DATA] Loaded {len(functional_flow)} timesteps ({hours[-1]:.1f} hours)")
print(f"[STAT:functional_mean] {np.mean(functional_flow):.2f}")
print(f"[STAT:functional_std] {np.std(functional_flow):.2f}")
print(f"[STAT:dead_mean] {np.mean(dead_flow):.2f}")

# Create synthetic noise examples
np.random.seed(42)
clean = functional_flow.copy()
node_std = np.std(clean[clean > 0]) if np.any(clean > 0) else 1.0

print(f"[STAT:node_std] {node_std:.2f}")

# 1. Gaussian noise (severity=0.5)
gaussian_noisy = clean.copy()
gaussian_noisy += np.random.normal(0, 0.5 * node_std, len(clean))
gaussian_noisy = np.maximum(gaussian_noisy, 0)

# 2. Bias (constant offset, severity=0.3)
bias_noisy = clean.copy()
bias_noisy += 0.3 * node_std

# 3. Drift (linearly increasing offset)
drift_noisy = clean.copy()
drift_factor = np.linspace(0, 0.5 * node_std, len(clean))
drift_noisy += drift_factor

# 4. Stuck (frozen at a single value)
stuck_noisy = clean.copy()
stuck_val = clean[96]  # value at hour 8
stuck_noisy[96:240] = stuck_val

print("[FINDING] Generated 4 noise types: Gaussian, Bias, Drift, Stuck")

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(14, 9), dpi=200)

# Shared y-axis range
y_max = np.max(clean) * 1.2

# Subplot titles
titles = [
    ("Functional Sensor", "Dead Sensor"),
    ("Gaussian Noise (σ=0.5)", "Bias (offset=0.3σ)"),
    ("Drift (0→0.5σ)", "Stuck (frozen value)")
]

# Row 1: Functional vs Dead
axes[0, 0].plot(hours, clean, color='steelblue', linewidth=1.5, alpha=0.9)
axes[0, 0].set_title(titles[0][0], fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0, y_max)
axes[0, 0].set_ylabel('Flow (veh/5min)', fontsize=10)

axes[0, 1].plot(hours, dead_flow, color='dimgray', linewidth=1.5, alpha=0.9)
axes[0, 1].set_title(titles[0][1] + " (Detectable)", fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(0, y_max)

# Row 2: Gaussian noise and Bias
axes[1, 0].plot(hours, clean, color='steelblue', linewidth=1.0, alpha=0.3, label='Clean')
axes[1, 0].plot(hours, gaussian_noisy, color='indianred', linewidth=1.5, alpha=0.9, label='Noisy')
axes[1, 0].set_title(titles[1][0], fontsize=12, fontweight='bold')
axes[1, 0].set_ylim(0, y_max)
axes[1, 0].set_ylabel('Flow (veh/5min)', fontsize=10)

axes[1, 1].plot(hours, clean, color='steelblue', linewidth=1.0, alpha=0.3)
axes[1, 1].plot(hours, bias_noisy, color='indianred', linewidth=1.5, alpha=0.9)
axes[1, 1].set_title(titles[1][1], fontsize=12, fontweight='bold')
axes[1, 1].set_ylim(0, y_max)

# Row 3: Drift and Stuck
axes[2, 0].plot(hours, clean, color='steelblue', linewidth=1.0, alpha=0.3)
axes[2, 0].plot(hours, drift_noisy, color='indianred', linewidth=1.5, alpha=0.9)
axes[2, 0].set_title(titles[2][0], fontsize=12, fontweight='bold')
axes[2, 0].set_ylim(0, y_max)
axes[2, 0].set_xlabel('Time (hours)', fontsize=10)
axes[2, 0].set_ylabel('Flow (veh/5min)', fontsize=10)

axes[2, 1].plot(hours, clean, color='steelblue', linewidth=1.0, alpha=0.3)
axes[2, 1].plot(hours, stuck_noisy, color='indianred', linewidth=1.5, alpha=0.9)
axes[2, 1].set_title(titles[2][1], fontsize=12, fontweight='bold')
axes[2, 1].set_ylim(0, y_max)
axes[2, 1].set_xlabel('Time (hours)', fontsize=10)

# X-ticks for all subplots
for ax in axes.flat:
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.grid(True, alpha=0.3)

# Add row labels for undetectable failures
fig.text(0.5, 0.37, 'Undetectable — values present but wrong',
         ha='center', fontsize=11, style='italic', color='darkred')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.25, bottom=0.08)

# Save figure
output_path = f'{output_dir}/sensor_failure_types.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"[FINDING] Figure saved to {output_path}")
print(f"[STAT:figure_size] (14, 9) inches at 200 DPI")
print(f"[STAT:y_max] {y_max:.2f}")
