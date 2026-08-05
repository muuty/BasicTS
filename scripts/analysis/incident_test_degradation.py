"""
Incident-case test-time performance degradation analysis.

Each trained run saved `test_incident_metrics.json` with MAE stratified by
whether the test sample covers an incident:
  non_incident_overall.MAE   — test samples with NO incident
  all_incident_overall.MAE   — test samples covering ANY severe incident
  {incident_type}_overall.MAE — per-type (Hazard, Collision, ...)

Plus horizon-split variants. We pair this with:
  - Coreset selection metadata (method, ratio, seed, distance_type) from cfg.txt
  - Training-set incident recall from incident_coverage_by_method.csv

Research question
-----------------
Do coreset methods with LOW training-set incident recall (recent, graph_cut)
produce trained models with LARGER MAE degradation on incident-containing
test cases than methods with HIGH recall (k_medoids, stride)?

Metric
------
  degradation_abs = MAE_incident - MAE_non_incident
  degradation_rel = (MAE_incident - MAE_non_incident) / MAE_non_incident

A NEGATIVE degradation means incident samples are actually EASIER for the
model (can happen — incidents often drop flow to near-zero which is trivially
predictable). What we care about is whether *relative* degradation differs
across coreset methods.

Inputs
------
- checkpoints/**/test_incident_metrics.json  (1641 runs)
- checkpoints/**/cfg.txt                     (to identify method/ratio/seed)
- experiments/result/analysis/incident_coverage_by_method.csv

Outputs (experiments/result/analysis/)
--------------------------------------
- incident_test_runs.csv            : one row per run (raw pull)
- incident_test_by_method.csv       : aggregated per (model, ds, method, ratio)
- incident_recall_vs_degradation.csv : joined with training-set recall

Usage
-----
  conda activate cuda && python scripts/analysis/incident_test_degradation.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / 'checkpoints'
OUTDIR = REPO / 'experiments/result/analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)

CFG_PAT_METHOD    = re.compile(r'SELECTION_STRATEGY:\s*(\S+)')
CFG_PAT_RATIO     = re.compile(r'SELECTION_RATIO:\s*([0-9.]+)')
CFG_PAT_SEED      = re.compile(r'\n\s*SEED:\s*(\d+)')  # first match — coreset seed
CFG_PAT_DISTANCE  = re.compile(r'DISTANCE_TYPE:\s*(\S+)')
CFG_PAT_DATASET   = re.compile(r"dataset_name:\s*xtraffic/(\S+)")


def parse_cfg(cfg_path: Path) -> dict:
    """Parse cfg.txt for coreset + dataset identifiers. Returns {} if not a
    coreset-trained run (full-data has no CORESET section)."""
    try:
        text = cfg_path.read_text()
    except Exception:
        return {}
    out = {}
    if (m := CFG_PAT_METHOD.search(text)):    out['method']   = m.group(1)
    if (m := CFG_PAT_RATIO.search(text)):     out['ratio']    = float(m.group(1))
    if (m := CFG_PAT_SEED.search(text)):      out['seed']     = int(m.group(1))
    if (m := CFG_PAT_DISTANCE.search(text)):  out['distance'] = m.group(1)
    if (m := CFG_PAT_DATASET.search(text)):   out['dataset']  = m.group(1)
    return out


def infer_model_from_path(p: Path) -> str | None:
    """checkpoints/<phase>/<model>/xtraffic/<ds>_<...>/<n>/<hash>/"""
    parts = p.relative_to(CKPT_ROOT).parts
    return parts[1] if len(parts) >= 2 else None


def collect_runs() -> pd.DataFrame:
    rows = []
    for inc_path in CKPT_ROOT.rglob('test_incident_metrics.json'):
        run_dir = inc_path.parent
        cfg = parse_cfg(run_dir / 'cfg.txt')
        if not cfg.get('method') or 'ratio' not in cfg:
            continue
        try:
            im = json.load(open(inc_path))
            tm = json.load(open(run_dir / 'test_metrics.json'))
        except Exception:
            continue
        # Pull the key metrics
        non_mae = im.get('non_incident_overall', {}).get('MAE')
        inc_mae = im.get('all_incident_overall',  {}).get('MAE')
        over_mae = tm.get('overall', {}).get('MAE')
        if None in (non_mae, inc_mae, over_mae) or non_mae == 0:
            continue
        rows.append({
            'model':    infer_model_from_path(run_dir),
            'dataset':  cfg.get('dataset'),
            'method':   cfg['method'],
            'ratio':    cfg['ratio'],
            'seed':     cfg.get('seed'),
            'distance': cfg.get('distance'),
            'phase':    run_dir.relative_to(CKPT_ROOT).parts[0],
            'overall_MAE':        round(over_mae, 4),
            'non_incident_MAE':   round(non_mae, 4),
            'all_incident_MAE':   round(inc_mae, 4),
            'degradation_abs':    round(inc_mae - non_mae, 4),
            'degradation_rel_pct': round(100 * (inc_mae - non_mae) / non_mae, 3),
            'run_dir': str(run_dir.relative_to(REPO)),
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average over seed + distance per (model, dataset, method, ratio)."""
    g = (df.groupby(['model', 'dataset', 'method', 'ratio'])
         .agg(n_runs=('seed', 'size'),
              overall_MAE=('overall_MAE', 'mean'),
              non_incident_MAE=('non_incident_MAE', 'mean'),
              all_incident_MAE=('all_incident_MAE', 'mean'),
              degradation_abs=('degradation_abs', 'mean'),
              degradation_rel_pct=('degradation_rel_pct', 'mean'),
              degradation_rel_std=('degradation_rel_pct', 'std'))
         .reset_index())
    return g.sort_values(['dataset', 'ratio', 'model', 'method'])


def join_with_training_recall(agg: pd.DataFrame) -> pd.DataFrame:
    """Add training-set incident recall (from incident_coverage_by_method.csv)
    to each row. Lets us correlate training recall with test degradation."""
    recall = pd.read_csv(OUTDIR / 'incident_coverage_by_method.csv')
    m = agg.merge(recall[['dataset', 'method', 'ratio', 'recall', 'enrichment']],
                  on=['dataset', 'method', 'ratio'], how='left')
    return m.sort_values(['dataset', 'ratio', 'model', 'method'])


def correlations(joined: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, ratio), Spearman between training recall and test
    degradation (both abs and rel)."""
    rows = []
    for r in sorted(joined['ratio'].unique()):
        for ds_label, sub in (
            [('pooled', joined[joined['ratio'] == r])] +
            [(ds, joined[(joined['ratio'] == r) & (joined['dataset'] == ds)])
             for ds in sorted(joined['dataset'].dropna().unique())]
        ):
            sub = sub.dropna(subset=['recall', 'degradation_rel_pct'])
            if len(sub) < 5:
                continue
            rho_abs, p_abs = spearmanr(sub['recall'], sub['degradation_abs'])
            rho_rel, p_rel = spearmanr(sub['recall'], sub['degradation_rel_pct'])
            rows.append({
                'ratio': r, 'dataset': ds_label, 'n': len(sub),
                'recall_vs_degradation_abs': round(rho_abs, 3),
                'p_abs': round(p_abs, 4),
                'recall_vs_degradation_rel': round(rho_rel, 3),
                'p_rel': round(p_rel, 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print('Scanning checkpoints ...')
    df = collect_runs()
    print(f'  Collected {len(df)} runs')
    print(f'  models:   {sorted(df["model"].dropna().unique())}')
    print(f'  datasets: {sorted(df["dataset"].dropna().unique())}')
    print(f'  methods:  {sorted(df["method"].unique())}')
    print(f'  ratios:   {sorted(df["ratio"].unique())}')
    df.to_csv(OUTDIR / 'incident_test_runs.csv', index=False)

    agg = aggregate(df)
    agg.to_csv(OUTDIR / 'incident_test_by_method.csv', index=False)

    joined = join_with_training_recall(agg)
    joined.to_csv(OUTDIR / 'incident_recall_vs_degradation.csv', index=False)

    print('\n=== Aggregated (model, dataset, method, ratio) — rel_pct degradation ===')
    # Pivot for readability
    pv = (agg.pivot_table(index=['dataset', 'ratio', 'method'],
                          columns='model', values='degradation_rel_pct',
                          aggfunc='mean')
          .round(2))
    print(pv.to_string())

    print('\n=== Correlation: training recall vs test degradation ===')
    corr = correlations(joined)
    corr.to_csv(OUTDIR / 'incident_recall_degradation_spearman.csv', index=False)
    print(corr.to_string(index=False))


if __name__ == '__main__':
    main()
