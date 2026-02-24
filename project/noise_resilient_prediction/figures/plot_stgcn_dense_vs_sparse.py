"""STGCN dense vs sparse adj comparison chart for PPT."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

# Data: 5 noise configs at r=30%
# (label, dense_baseline, dense_v2, sparse_baseline, sparse_v2)
configs = [
    ('Clean',              13.34, 15.08, 13.99, 14.03),
    ('Gaussian\n(mild)',   15.10, 15.51, 16.48, 14.83),
    ('Gaussian\n(moderate)', 18.39, 16.11, 21.59, 15.73),
    ('Gaussian\n(severe)', 27.77, 17.72, 40.74, 17.79),
    ('Bias',               19.30, 18.81, 19.85, 17.78),
    ('Drift',              17.99, 17.53, 18.04, 16.55),
]

labels = [c[0] for c in configs]
dense_base = [c[1] for c in configs]
dense_v2   = [c[2] for c in configs]
sparse_base = [c[3] for c in configs]
sparse_v2   = [c[4] for c in configs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

x = np.arange(len(labels))
width = 0.35

# --- Left: Dense Adj ---
bars1 = ax1.bar(x - width/2, dense_base, width, label='Baseline',
                color='#9ECAE1', edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax1.bar(x + width/2, dense_v2, width, label='+ v2 Combo',
                color='#2171B5', edgecolor='white', linewidth=0.5, zorder=3)

for bar, val in zip(bars1, dense_base):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, color='#555555', fontweight='bold')
for i, (bar, val) in enumerate(zip(bars2, dense_v2)):
    diff = val - dense_base[i]
    sign = '+' if diff >= 0 else ''
    color = '#CC0000' if diff > 0.5 else '#006600'
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f'{val:.1f}\n({sign}{diff:.1f})', ha='center', va='bottom', fontsize=8.5,
             color=color, fontweight='bold')

ax1.set_title('Dense Adjacency (95% density)\nClean penalty: +13%', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('MAE (Flow)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Right: Sparse Adj ---
bars3 = ax2.bar(x - width/2, sparse_base, width, label='Baseline',
                color='#FDAE6B', edgecolor='white', linewidth=0.5, zorder=3)
bars4 = ax2.bar(x + width/2, sparse_v2, width, label='+ v2 Combo',
                color='#D94801', edgecolor='white', linewidth=0.5, zorder=3)

for bar, val in zip(bars3, sparse_base):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, color='#555555', fontweight='bold')
for i, (bar, val) in enumerate(zip(bars4, sparse_v2)):
    diff = val - sparse_base[i]
    sign = '+' if diff >= 0 else ''
    color = '#CC0000' if diff > 0.5 else '#006600'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f'{val:.1f}\n({sign}{diff:.1f})', ha='center', va='bottom', fontsize=8.5,
             color=color, fontweight='bold')

ax2.set_title('Sparse Adjacency (28.5% density)\nClean penalty: +0.3%', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Shared y-axis
ax1.set_ylim(0, max(max(sparse_base), max(dense_base)) * 1.22)

fig.suptitle('STGCN: Effect of Graph Topology on Encoder Integration',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('project/noise_resilient_prediction/figures/stgcn_dense_vs_sparse.png', dpi=200, bbox_inches='tight')
plt.savefig('project/noise_resilient_prediction/figures/stgcn_dense_vs_sparse.pdf', bbox_inches='tight')
print("Saved.")
