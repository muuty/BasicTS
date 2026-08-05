"""
Two coreset-selection characterizations needed to firm up the paper story:

  (Q2) Performance recovery vs full data
       At r ∈ {0.5, 0.7, 0.9}, does the best coreset match (or beat) the
       ratio=1.0 baseline? Quantify the gap per (model, dataset).

  (Q3) Who beats k_medoids at low ratio?
       At r ∈ {0.1, 0.3}, k_medoids is qc-optimal. Enumerate every
       (model, dataset) where another method achieves lower MAE, and by how
       much. Helps decide whether the "k_medoids is theoretically justified"
       story has exceptions we need to explain.

Inputs
------
- experiments/result/unified_mae_with_std.csv : seed-averaged MAE per
  (model, dataset, method, ratio). ratio=1.0 is the full-data baseline.

Outputs (experiments/result/analysis/)
--------------------------------------
- coreset_recovery_vs_full.csv   : per (model, dataset, ratio), best coreset
                                    MAE vs full MAE, and which method won.
- coreset_recovery_summary.csv   : how often does best-coreset ≤ full at each
                                    ratio, pooled across (model, dataset).
- kmedoids_losses_low_ratio.csv  : every (model, dataset, ratio ≤ 0.3) where
                                    another method beats k_medoids, with gap.

Usage
-----
  conda activate cuda && python scripts/analysis/coreset_performance_characterization.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
UNIFIED = REPO / 'experiments/result/unified_mae_with_std.csv'
OUTDIR = REPO / 'experiments/result/analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)


def recovery_vs_full(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (model, dataset), compare best coreset MAE at r<1 to r=1.0.

    Reports:
      - abs_gap  = best_coreset_MAE - full_MAE
      - rel_gap  = abs_gap / full_MAE
      - beats_full = (abs_gap < 0)
    """
    full = (df[df['ratio'] == 1.0]
            .groupby(['model', 'dataset'])['MAE_mean'].mean()
            .rename('full_MAE').reset_index())
    subs = df[df['ratio'] < 1.0]
    best = (subs.loc[subs.groupby(['model', 'dataset', 'ratio'])['MAE_mean'].idxmin()]
            [['model', 'dataset', 'ratio', 'method', 'MAE_mean']]
            .rename(columns={'method': 'best_method', 'MAE_mean': 'best_MAE'}))
    merged = best.merge(full, on=['model', 'dataset'])
    merged['abs_gap']    = merged['best_MAE'] - merged['full_MAE']
    merged['rel_gap_pct'] = 100 * merged['abs_gap'] / merged['full_MAE']
    merged['beats_full'] = merged['abs_gap'] < 0
    merged = merged.sort_values(['ratio', 'dataset', 'model'])

    # Pooled summary per ratio
    summary = (merged.groupby('ratio')
               .agg(n=('model', 'size'),
                    beats_full=('beats_full', 'sum'),
                    mean_abs_gap=('abs_gap', 'mean'),
                    mean_rel_gap_pct=('rel_gap_pct', 'mean'),
                    median_rel_gap_pct=('rel_gap_pct', 'median'),
                    max_rel_gap_pct=('rel_gap_pct', 'max'))
               .reset_index())
    return merged, summary


def kmedoids_losses(df: pd.DataFrame, low_ratios=(0.1, 0.3)) -> pd.DataFrame:
    """For each (model, dataset, ratio in low_ratios), report every method
    that beats k_medoids MAE, with the gap."""
    rows = []
    low = df[df['ratio'].isin(low_ratios)]
    for (model, dataset, ratio), g in low.groupby(['model', 'dataset', 'ratio']):
        km = g[g['method'] == 'k_medoids']
        if km.empty:
            continue
        km_mae = km.iloc[0]['MAE_mean']
        for _, r in g.iterrows():
            if r['method'] == 'k_medoids':
                continue
            if r['MAE_mean'] < km_mae:
                rows.append({
                    'model': model, 'dataset': dataset, 'ratio': ratio,
                    'winner': r['method'], 'winner_MAE': round(r['MAE_mean'], 3),
                    'k_medoids_MAE': round(km_mae, 3),
                    'abs_gap': round(r['MAE_mean'] - km_mae, 3),
                    'rel_gap_pct': round(100 * (r['MAE_mean'] - km_mae) / km_mae, 2),
                })
    return (pd.DataFrame(rows)
            .sort_values(['ratio', 'dataset', 'model', 'rel_gap_pct'])
            if rows else pd.DataFrame(columns=['model','dataset','ratio','winner']))


def main() -> None:
    df = pd.read_csv(UNIFIED)
    print(f'Loaded {len(df)} rows  '
          f'(models={df["model"].nunique()}, datasets={df["dataset"].nunique()}, '
          f'ratios={sorted(df["ratio"].unique())}, methods={df["method"].nunique()})')

    recovery, summary = recovery_vs_full(df)
    recovery.to_csv(OUTDIR / 'coreset_recovery_vs_full.csv', index=False)
    summary.to_csv(OUTDIR / 'coreset_recovery_summary.csv', index=False)
    print('\n--- coreset_recovery_summary.csv ---')
    print(summary.to_string(index=False))
    print('\n--- coreset_recovery_vs_full.csv (best method + gap per (model, ds, ratio)) ---')
    print(recovery.to_string(index=False))

    losses = kmedoids_losses(df)
    losses.to_csv(OUTDIR / 'kmedoids_losses_low_ratio.csv', index=False)
    print('\n--- kmedoids_losses_low_ratio.csv (cases where another method beats k_medoids at r∈{0.1,0.3}) ---')
    if losses.empty:
        print('  (none — k_medoids is the best method in every low-ratio cell)')
    else:
        print(losses.to_string(index=False))
        # Aggregate which methods win and how often
        print('\nWinner counts (who beats k_medoids, and by how much on average):')
        agg = (losses.groupby(['ratio', 'winner'])
               .agg(n=('model', 'size'),
                    mean_rel_gap_pct=('rel_gap_pct', 'mean'))
               .reset_index()
               .sort_values(['ratio', 'n'], ascending=[True, False]))
        print(agg.to_string(index=False))


if __name__ == '__main__':
    main()
