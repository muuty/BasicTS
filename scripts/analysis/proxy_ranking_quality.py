"""
Proxy ranking quality analysis — correct qc variants.

Compares multiple qc formulations against MAE to answer:
  (1) Per-ratio Spearman rank correlation — qc_raw_l1 is the MAE-aligned
      upper bound from Kantorovich duality and should be the primary qc.
  (2) Per-ratio top-1 hit rate: does best-qc pick = best-MAE method?
  (3) Per-ratio qc ranking by method — which method is qc #1 at each ratio?

qc variants compared
--------------------
- qc_raw_l1          : (1/n) Σ min_{s∈S} ||x-s||_1   [train → coreset, raw, L1]
                       theoretically correct: coreset bound is MAE ≤ MAE_S + L·qc_raw_l1
- qc_test_l1         : (1/n_test) Σ min_{s∈S} ||x_test - s||_1  [test → coreset]
- qc_reverse_l1      : (1/|S|) Σ min_{x_test} ||s - x_test||_1  [coreset → test]
- qc_bidir_l1        : qc_test_l1 + qc_reverse_l1  (W1 upper bound, symmetric)
- ot_cost            : Sinkhorn regularized OT, PCA-50, squared Euclidean (W2² proxy)
                       INCLUDED for reference only — aligns with MSE, not MAE,
                       and produces artifactual near-zero values for contiguous
                       selections (e.g. `recent`).
- h_tod, h_dow       : normalized entropy of time-of-day / day-of-week bins
- h_td_avg           : (h_tod + h_dow) / 2

Inputs
------
- experiments/result/unified_mae_with_std.csv : seed-averaged MAE per
  (model, dataset, method, ratio).
- coreset_indices/{DATASET}/proxy_metrics.json : all qc variants per
  (strategy, distance, ratio, seed). 117/206 entries have qc_raw_l1 as of
  this analysis; missing rows drop out of their respective analyses.

Outputs (experiments/result/analysis/)
--------------------------------------
- proxy_spearman_by_ratio.csv     : per-ratio Spearman for every qc variant
- proxy_spearman_by_tod_usage.csv : Spearman stratified by whether the model
                                     uses TOD/DOW embedding (STID/STAEformer)
- proxy_spearman_by_model.csv     : per-model within-(ratio,dataset) Spearman
- proxy_top1_hits_by_ratio.csv    : top-1 hit rate per ratio
- proxy_qc_ranking_by_method.csv  : method-level qc values per (dataset,ratio)

Usage
-----
  conda activate cuda && python scripts/analysis/proxy_ranking_quality.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
UNIFIED_MAE = REPO / 'experiments/result/unified_mae_with_std.csv'
PROXY_FILES = {
    'SAN_BERNARDINO': REPO / 'coreset_indices/SAN_BERNARDINO/proxy_metrics.json',
    'CONTRA_COSTA':   REPO / 'coreset_indices/CONTRA_COSTA/proxy_metrics.json',
}
OUTDIR = REPO / 'experiments/result/analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)

TOD_USING_MODELS = {'STID', 'STAEformer'}
VARIANT_RE = r'_v2|_cosine|_old'

# Proxy columns to evaluate. "lower is better" vs "higher is better" is encoded
# here so downstream ranking logic is consistent.
PROXY_SPEC = [
    ('qc_raw_l1',       'lower'),
    ('qc_test_l1',      'lower'),
    ('qc_reverse_l1',   'lower'),
    ('qc_bidir_l1',     'lower'),
    ('ot_cost',         'lower'),
    ('h_tod',           'higher'),
    ('h_dow',           'higher'),
    ('h_td_avg',        'higher'),
]
PROXY_COLS = [p for p, _ in PROXY_SPEC]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_proxy_table(path: Path, dataset: str) -> pd.DataFrame:
    pm = json.load(open(path))
    rows = []
    for fname, v in pm.items():
        name = fname.replace('.json', '')
        parts = name.split('_')
        seed_idx = next(i for i, p in enumerate(parts) if p.startswith('seed'))
        qc_test = v.get('qc_test_l1')
        qc_rev  = v.get('qc_reverse_l1')
        qc_bidir = (qc_test + qc_rev) if (qc_test is not None and qc_rev is not None) else None
        rows.append({
            'dataset':   dataset,
            'method':    '_'.join(parts[:seed_idx - 2]),
            'distance':  parts[seed_idx - 2],
            'ratio':     int(parts[seed_idx - 1]) / 100,
            'seed':      int(parts[seed_idx].replace('seed', '')),
            'qc_raw_l1':     v.get('qc_raw_l1'),
            'qc_test_l1':    qc_test,
            'qc_reverse_l1': qc_rev,
            'qc_bidir_l1':   qc_bidir,
            'ot_cost':       v.get('ot_cost'),
            'h_tod':         v.get('h_tod'),
            'h_dow':         v.get('h_dow'),
        })
    return pd.DataFrame(rows)


def load_merged() -> pd.DataFrame:
    proxy = pd.concat([load_proxy_table(p, ds) for ds, p in PROXY_FILES.items()])
    proxy = proxy[~proxy['method'].str.contains(VARIANT_RE, regex=True)]

    # unified_mae is averaged over seed+distance already. Aggregate proxy the
    # same way. NaN propagates (some methods lack some qc variants).
    proxy_agg = (proxy.groupby(['dataset', 'method', 'ratio'])
                 .agg(**{col: (col, 'mean') for col in
                         ['qc_raw_l1', 'qc_test_l1', 'qc_reverse_l1',
                          'qc_bidir_l1', 'ot_cost', 'h_tod', 'h_dow']})
                 .reset_index())

    mae = pd.read_csv(UNIFIED_MAE)
    mae = mae[mae['ratio'] < 1.0]

    m = mae.merge(proxy_agg, on=['dataset', 'method', 'ratio'], how='inner')
    m['uses_tod'] = m['model'].isin(TOD_USING_MODELS)
    m['h_td_avg'] = (m['h_tod'] + m['h_dow']) / 2
    return m


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _spearman_dict(sub: pd.DataFrame) -> dict:
    """Return {proxy: rho} per proxy, skipping if <4 non-null pairs."""
    out = {'n': len(sub)}
    for p in PROXY_COLS:
        s = sub[['MAE_mean', p]].dropna()
        if len(s) < 4:
            out[p] = np.nan
            continue
        rho, _ = spearmanr(s[p], s['MAE_mean'])
        out[p] = round(rho, 3)
    return out


def spearman_by_ratio(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in sorted(m['ratio'].unique()):
        for label, sub in [('all',       m[m['ratio'] == r]),
                           ('no_recent', m[(m['ratio'] == r) & (m['method'] != 'recent')])]:
            rows.append({'ratio': r, 'set': label, **_spearman_dict(sub)})
    return pd.DataFrame(rows)


def spearman_by_tod_usage(m: pd.DataFrame) -> pd.DataFrame:
    """STID/STAEformer vs others. `recent` kept — qc_raw_l1 handles it correctly
    (unlike ot_cost)."""
    rows = []
    for r in sorted(m['ratio'].unique()):
        for uses in [True, False]:
            sub = m[(m['ratio'] == r) & (m['uses_tod'] == uses)]
            rows.append({
                'ratio': r,
                'group': 'STID_STAEformer' if uses else 'AGCRN_DCRNN_STGCN',
                **_spearman_dict(sub),
            })
    return pd.DataFrame(rows)


def spearman_by_model_within_group(m: pd.DataFrame) -> pd.DataFrame:
    """Within-(ratio,dataset) rank correlation per model, pooled across groups."""
    rows = []
    for model, sub in m.groupby('model'):
        sub = sub.copy()
        for p in PROXY_COLS + ['MAE_mean']:
            sub[f'{p}_rank'] = sub.groupby(['ratio', 'dataset'])[p].rank()
        row = {'model': model, 'uses_tod': model in TOD_USING_MODELS, 'n': len(sub)}
        for p in PROXY_COLS:
            s = sub[[f'{p}_rank', 'MAE_mean_rank']].dropna()
            if len(s) < 4:
                row[p] = np.nan
                continue
            rho, _ = spearmanr(s[f'{p}_rank'], s['MAE_mean_rank'])
            row[p] = round(rho, 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('model')


def top1_hits_by_ratio(m: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset, ratio), does the method with best proxy value
    match the method with best (lowest) MAE?"""
    rows = []
    for r in sorted(m['ratio'].unique()):
        counts = {p: [0, 0] for p, _ in PROXY_SPEC}   # [hits, n]
        for (ds, md), sub in m[m['ratio'] == r].groupby(['dataset', 'model']):
            if len(sub) < 3:
                continue
            best_mae = sub.sort_values('MAE_mean').iloc[0]['method']
            for p, direction in PROXY_SPEC:
                s = sub.dropna(subset=[p])
                if len(s) < 3:
                    continue
                ascending = (direction == 'lower')
                pick = s.sort_values(p, ascending=ascending).iloc[0]['method']
                counts[p][0] += (best_mae == pick)
                counts[p][1] += 1
        row = {'ratio': r}
        for p, _ in PROXY_SPEC:
            hits, n = counts[p]
            row[f'{p}_hit'] = f'{hits}/{n}' if n else 'na'
        rows.append(row)
    return pd.DataFrame(rows)


def qc_ranking_by_method(m: pd.DataFrame, qc_col: str = 'qc_raw_l1') -> pd.DataFrame:
    """Method-level qc_raw_l1 values per (dataset, ratio), with within-group rank."""
    sub = m.groupby(['dataset', 'ratio', 'method'])[qc_col].mean().reset_index()
    sub['qc_rank'] = sub.groupby(['dataset', 'ratio'])[qc_col].rank().astype('Int64')
    return sub.sort_values(['dataset', 'ratio', 'qc_rank'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    m = load_merged()
    cov_qc_raw = m['qc_raw_l1'].notna().sum()
    cov_qc_test = m['qc_test_l1'].notna().sum()
    print(f'Loaded {len(m)} merged rows '
          f'({m["model"].nunique()} models × {m["dataset"].nunique()} ds × '
          f'{m["method"].nunique()} methods × {m["ratio"].nunique()} ratios)')
    print(f'Coverage: qc_raw_l1 {cov_qc_raw}/{len(m)}, '
          f'qc_test_l1 {cov_qc_test}/{len(m)}')

    outputs = {
        'proxy_spearman_by_ratio.csv':     spearman_by_ratio(m),
        'proxy_spearman_by_tod_usage.csv': spearman_by_tod_usage(m),
        'proxy_spearman_by_model.csv':     spearman_by_model_within_group(m),
        'proxy_top1_hits_by_ratio.csv':    top1_hits_by_ratio(m),
        'proxy_qc_ranking_by_method.csv':  qc_ranking_by_method(m, 'qc_raw_l1'),
    }
    for name, df in outputs.items():
        path = OUTDIR / name
        df.to_csv(path, index=False)
        print(f'\n--- {name} ({len(df)} rows) ---')
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
