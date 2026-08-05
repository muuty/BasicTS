#!/usr/bin/env python3
"""Two-panel figure: within-ratio ranking + noise floor.

Panel (a): within-ratio |Spearman ρ| of proxy vs MAE — shows quant_cost
           dominates the differentiating regime.
Panel (b): method/selection-instantiation separation by ratio.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'experiments/result/analysis'
FIG = ROOT / 'writing/CoresetSelection-paper/figures'

noise = pd.read_csv(OUT / 'proxy_diag1_noise_floor.csv')
spearman = pd.read_csv(OUT / 'proxy_diag2_per_ratio_spearman.csv')

ratio_col = 'separation_ratio' if 'separation_ratio' in noise else 'snr'
snr_med = noise.groupby('ratio')[ratio_col].median().reset_index()

palette = {
    'quantization_cost':    ('tab:blue',   '-',  'o', 'Quantization cost'),
    'sinkhorn_divergence':  ('tab:orange', '--', 's', 'Sinkhorn divergence'),
    'ot_cost':              ('tab:green',  '--', '^', 'OT cost'),
    'fl_objective':         ('tab:red',    ':',  'D', 'Facility location'),
    'h_tod':                ('tab:purple', ':',  'v', 'Time-of-day entropy'),
}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

# Panel (a): within-ratio |ρ|
ax = axes[0]
for pm, (color, ls, marker, label) in palette.items():
    sub = spearman[spearman['proxy_metric'] == pm].dropna(subset=['median_rho'])
    if len(sub) == 0:
        continue
    ax.plot(sub['ratio'], sub['median_rho'].abs(),
            ls=ls, marker=marker, color=color, label=label, lw=1.5)
ax.set_xlabel('Sampling ratio')
ax.set_ylabel(r'Within-ratio $|\rho_{\mathrm{Spearman}}|$  (median across 5 backbones)')
ax.set_title('(a) Proxy ranking power, holding ratio fixed')
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc='upper right')

# Panel (b): noise floor SNR
ax = axes[1]
ax.plot(snr_med['ratio'], snr_med[ratio_col], '-o', color='black', lw=2)
ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.axhline(2, color='gray', ls=':',  lw=0.8, alpha=0.6)
ax.set_yscale('log')
ax.set_xlabel('Sampling ratio')
ax.set_ylabel(r'$\sigma_{\mathrm{method}}\,/\,\sigma_{\mathrm{selection}}$')
ax.set_title('(b) Method/selection-instantiation separation')
ax.text(0.92, 1.05, 'ratio$=$1: equal observed spreads',
        transform=ax.get_yaxis_transform(), fontsize=8, color='gray',
        ha='right')
ax.text(0.92, 2.05, 'ratio$=$2: descriptive reference',
        transform=ax.get_yaxis_transform(), fontsize=8, color='gray',
        ha='right')
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIG / 'proxy_diagnostics.pdf', dpi=150, bbox_inches='tight')
fig.savefig(FIG / 'proxy_diagnostics.png', dpi=150, bbox_inches='tight')
print(f'Saved → {FIG}/proxy_diagnostics.{{pdf,png}}')
