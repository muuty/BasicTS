"""Healthy-only MAE: Baseline vs Ours for both architectures."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

# --- Data (Healthy nodes only, r=30%) ---
# From noise vulnerability eval output
gauss_labels = ['Clean', 'Mild\n(s=0.3)', 'Moderate\n(s=0.5)', 'Severe\n(s=1.0)']
gauss_stae_base  = [12.13, 14.63, 30.91, 66.53]
gauss_stae_ours  = [12.81, 12.83, 12.81, 12.88]
gauss_stgcn_base = [13.99, 15.21, 19.32, 22.76]  # sparse adj baseline (healthy=all at clean)
gauss_stgcn_ours = [14.03, 14.16, 14.38, 15.08]

bd_labels = ['Clean', 'Bias\n(s=0.3)', 'Drift\n(s=0.3)']
bd_stae_base  = [12.13, 15.25, 12.39]
bd_stae_ours  = [12.81, 12.93, 12.90]
bd_stgcn_base = [13.99, 14.58, 14.73]  # sparse adj baseline (healthy = from noise eval, need actual values)
bd_stgcn_ours = [14.03, 14.16, 14.13]

# Colors
C_STAE_BASE  = '#9ECAE1'
C_STAE_OURS  = '#2171B5'
C_STGCN_BASE = '#FDAE6B'
C_STGCN_OURS = '#D94801'

def annotate_bars(ax, bars, vals, ref_vals=None, fontsize=8.5):
    for i, (bar, val) in enumerate(zip(bars, vals)):
        if ref_vals and i > 0:
            diff = val - ref_vals[i]
            txt = f'{val:.1f}\n({diff:+.1f})'
            color = '#006600' if diff < -0.3 else ('#CC0000' if diff > 0.3 else '#555555')
        else:
            txt = f'{val:.1f}'
            color = '#555555'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                txt, ha='center', va='bottom', fontsize=fontsize, color=color, fontweight='bold')

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
width = 0.2
all_vals = gauss_stae_base + gauss_stae_ours + gauss_stgcn_base + gauss_stgcn_ours + bd_stae_base + bd_stae_ours + bd_stgcn_base + bd_stgcn_ours
global_ylim = (0, max(all_vals) * 1.22)

# Top-left: STAEformer — Gaussian
ax = axes[0, 0]
x = np.arange(len(gauss_labels))
b1 = ax.bar(x - width/2, gauss_stae_base, width, label='Baseline', color=C_STAE_BASE, edgecolor='white', zorder=3)
b2 = ax.bar(x + width/2, gauss_stae_ours, width, label='Ours', color=C_STAE_OURS, edgecolor='white', zorder=3)
annotate_bars(ax, b1, gauss_stae_base)
annotate_bars(ax, b2, gauss_stae_ours, gauss_stae_base)
ax.set_title('STAEformer — Gaussian Noise', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE (Flow)', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(gauss_labels)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(global_ylim)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Top-right: STGCN — Gaussian
ax = axes[0, 1]
x = np.arange(len(gauss_labels))
b1 = ax.bar(x - width/2, gauss_stgcn_base, width, label='Baseline', color=C_STGCN_BASE, edgecolor='white', zorder=3)
b2 = ax.bar(x + width/2, gauss_stgcn_ours, width, label='Ours', color=C_STGCN_OURS, edgecolor='white', zorder=3)
annotate_bars(ax, b1, gauss_stgcn_base)
annotate_bars(ax, b2, gauss_stgcn_ours, gauss_stgcn_base)
ax.set_title('STGCN (Sparse Adj) — Gaussian Noise', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(gauss_labels)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(global_ylim)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Bottom-left: STAEformer — Bias & Drift
ax = axes[1, 0]
x = np.arange(len(bd_labels))
b1 = ax.bar(x - width/2, bd_stae_base, width, label='Baseline', color=C_STAE_BASE, edgecolor='white', zorder=3)
b2 = ax.bar(x + width/2, bd_stae_ours, width, label='Ours', color=C_STAE_OURS, edgecolor='white', zorder=3)
annotate_bars(ax, b1, bd_stae_base)
annotate_bars(ax, b2, bd_stae_ours, bd_stae_base)
ax.set_title('STAEformer — Bias & Drift Noise', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE (Flow)', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(bd_labels)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(global_ylim)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Bottom-right: STGCN — Bias & Drift
ax = axes[1, 1]
x = np.arange(len(bd_labels))
b1 = ax.bar(x - width/2, bd_stgcn_base, width, label='Baseline', color=C_STGCN_BASE, edgecolor='white', zorder=3)
b2 = ax.bar(x + width/2, bd_stgcn_ours, width, label='Ours', color=C_STGCN_OURS, edgecolor='white', zorder=3)
annotate_bars(ax, b1, bd_stgcn_base)
annotate_bars(ax, b2, bd_stgcn_ours, bd_stgcn_base)
ax.set_title('STGCN (Sparse Adj) — Bias & Drift Noise', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(bd_labels)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(global_ylim)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle('Healthy Node MAE Under Sensor Noise (Spillover Effect)\n(30% of sensors corrupted — measured on the remaining 70% healthy sensors)',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('project/noise_resilient_prediction/figures/healthy_only_by_noise_type.png', dpi=200, bbox_inches='tight')
plt.savefig('project/noise_resilient_prediction/figures/healthy_only_by_noise_type.pdf', bbox_inches='tight')
print("Saved.")
