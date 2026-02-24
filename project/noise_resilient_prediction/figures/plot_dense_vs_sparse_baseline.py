"""Dense vs Sparse baseline vulnerability comparison (no encoder, no augmentation)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

# Data: STGCN baseline (no defense), All Functional MAE
# Dense = STGCN 1ch baseline (stgcn_1ch from noise eval, clean=13.34)
# Sparse = STGCN 5ch sparse adj baseline (stgcn_5ch_sparse_adj, clean=13.99)
labels = ['Clean', 'Gaussian\n(mild)', 'Gaussian\n(moderate)', 'Gaussian\n(severe)', 'Bias', 'Drift']

dense_vals  = [13.34, 15.10, 18.39, 27.77, 19.30, 17.99]
sparse_vals = [13.99, 16.48, 21.59, 40.74, 19.85, 18.04]

dense_clean = 13.34
sparse_clean = 13.99

C_DENSE  = '#7570B3'
C_SPARSE = '#E7298A'

fig, ax = plt.subplots(figsize=(14, 6.5))

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, dense_vals, width, label='Dense Adj (95% density)',
               color=C_DENSE, edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax.bar(x + width/2, sparse_vals, width, label='Sparse Adj (28.5% density)',
               color=C_SPARSE, edgecolor='white', linewidth=0.5, zorder=3)

# Annotations
for i, (bar, val) in enumerate(zip(bars1, dense_vals)):
    if i > 0:
        pct = (val - dense_clean) / dense_clean * 100
        txt = f'{val:.1f}\n(+{pct:.0f}%)'
    else:
        txt = f'{val:.1f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            txt, ha='center', va='bottom', fontsize=9, color='#333333', fontweight='bold')

for i, (bar, val) in enumerate(zip(bars2, sparse_vals)):
    if i > 0:
        pct = (val - sparse_clean) / sparse_clean * 100
        txt = f'{val:.1f}\n(+{pct:.0f}%)'
    else:
        txt = f'{val:.1f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            txt, ha='center', va='bottom', fontsize=9, color='#333333', fontweight='bold')

ax.set_ylabel('MAE (Flow)', fontsize=13, fontweight='bold')
ax.set_title('STGCN Baseline Noise Vulnerability: Dense vs. Sparse Adjacency\n(30% of sensors corrupted)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.set_ylim(0, max(max(dense_vals), max(sparse_vals)) * 1.22)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('project/noise_resilient_prediction/figures/dense_vs_sparse_baseline.png', dpi=200, bbox_inches='tight')
plt.savefig('project/noise_resilient_prediction/figures/dense_vs_sparse_baseline.pdf', bbox_inches='tight')
print("Saved.")
