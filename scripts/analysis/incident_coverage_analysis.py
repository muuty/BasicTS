"""
Incident coverage analysis for coreset selection methods (SB + CC).

Traffic-specific question: when we prune training data, do coreset methods
preserve samples that contain severe incidents (flow z-score < -2), and does
incident coverage predict downstream MAE?

Definitions
-----------
- A *severe incident* is a (time, sensor) event where observed flow dropped
  > 2σ below the TOD baseline (from EDA/severe_incidents_{DATASET}.csv,
  produced by scripts/analysis/extract_severe_incidents.py).
- A training *sample* at index `i` has input window [i, i+12) and output
  window [i+12, i+24). We say sample `i` **covers** an incident if the
  incident_slot lies in either window, i.e. i ≤ incident_slot < i + 24.
- *Incident recall* = fraction of in-window incidents covered by ≥1 selected
  sample. Range [0, 1].
- *Incident density* = (incident-covering samples in coreset) / |coreset|.
  Higher than the base rate (incident samples / train size) ⇒ method is
  OVER-selecting incidents. We report the ratio density / base_rate as
  `incident_enrichment` (1.0 = proportional, >1 = enriched, <1 = depleted).

Inputs
------
- EDA/severe_incidents_{DATASET}.csv : severe incident metadata.
- coreset_indices/{DATASET}/*.json   : selection index lists.
- experiments/result/unified_mae_with_std.csv : MAE per (model, ds, method, r).

The training window is derived from the config: data_range=(0, 24192),
TRAIN_VAL_TEST_RATIO=[0.6, 0.2, 0.2], INPUT_LEN=12, OUTPUT_LEN=12.

Outputs (experiments/result/analysis/)
--------------------------------------
- incident_coverage_by_config.csv   : per-(method, distance, ratio, seed)
                                       recall + enrichment
- incident_coverage_by_method.csv   : averaged over seed+distance per
                                       (method, ratio) — main table for paper
- incident_coverage_vs_mae.csv      : joined with MAE; Spearman per ratio
                                       between recall/enrichment and MAE

Usage
-----
  conda activate cuda && python scripts/analysis/incident_coverage_analysis.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
UNIFIED_MAE = REPO / 'experiments/result/unified_mae_with_std.csv'
OUTDIR = REPO / 'experiments/result/analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['SAN_BERNARDINO', 'CONTRA_COSTA']


def incidents_csv(dataset: str) -> Path:
    return REPO / f'EDA/severe_incidents_{dataset}.csv'


def index_dir(dataset: str) -> Path:
    return REPO / f'coreset_indices/{dataset}'

# Config-derived constants (SAN_BERNARDINO, 3-month window)
INPUT_LEN  = 12
OUTPUT_LEN = 12
DATA_RANGE_END = 24192
VALID_RATIO = 0.2
TEST_RATIO = 0.2
# BasicTS splits timesteps first, then forms windows inside the training slice.
TRAIN_TIMESTEPS = (
    DATA_RANGE_END
    - int(DATA_RANGE_END * VALID_RATIO)
    - int(DATA_RANGE_END * TEST_RATIO)
)
N_TRAIN = TRAIN_TIMESTEPS - INPUT_LEN - OUTPUT_LEN + 1

FNAME_RE = re.compile(
    r'^(?P<method>.+?)_(?P<distance>euclidean|temporal|spatial|combined|'
    r'cosine_raw|cosine_temporal|cosine_spatial|cosine_combined)'
    r'_(?P<ratio>\d{3})_seed(?P<seed>\d+)$'
)
VARIANT_RE = re.compile(r'_v2|_cosine|_old')


def load_train_incident_starts(dataset: str) -> np.ndarray:
    """Return sorted array of sample-start indices `i` that cover at least
    one in-train severe incident. We only count incidents whose incident_slot
    falls inside the training index range."""
    inc = pd.read_csv(incidents_csv(dataset))
    covering_starts = set()
    for s in inc['incident_slot']:
        lo = max(0, s - (INPUT_LEN + OUTPUT_LEN) + 1)
        hi = min(N_TRAIN, s + 1)
        covering_starts.update(range(lo, hi))
    return np.array(sorted(covering_starts), dtype=np.int64)


def load_incidents_by_id(dataset: str) -> dict:
    """Map incident_id → set of sample-start indices that cover it.
    Multiple sensors can trigger at the same slot; we use (id, sensor_idx)
    as the key so same-slot different-sensor incidents count separately."""
    inc = pd.read_csv(incidents_csv(dataset))
    out = {}
    for _, r in inc.iterrows():
        s = int(r['incident_slot'])
        lo = max(0, s - (INPUT_LEN + OUTPUT_LEN) + 1)
        hi = min(N_TRAIN, s + 1)
        if lo < hi:
            key = (int(r['incident_id']), int(r['sensor_idx']))
            out[key] = set(range(lo, hi))
    return out


def coreset_metrics(selected: np.ndarray,
                    incident_starts: np.ndarray,
                    incidents_by_id: dict,
                    base_rate: float) -> dict:
    """Compute recall + density + enrichment for one coreset."""
    selected_set = set(int(x) for x in selected if 0 <= x < N_TRAIN)
    if not selected_set:
        return dict(recall=np.nan, density=np.nan, enrichment=np.nan,
                    n_train_samples=0, n_covered_incidents=0)
    # Recall: fraction of in-train incidents with ≥1 covering sample selected
    n_incidents = len(incidents_by_id)
    if n_incidents == 0:
        recall = np.nan
    else:
        covered = sum(1 for starts in incidents_by_id.values()
                      if starts & selected_set)
        recall = covered / n_incidents
    # Density: fraction of selected samples that cover any incident
    incident_sample_set = set(int(x) for x in incident_starts)
    n_inc_in_cs = len(selected_set & incident_sample_set)
    density = n_inc_in_cs / len(selected_set)
    enrichment = density / base_rate if base_rate > 0 else np.nan
    return dict(
        recall=round(recall, 4) if recall == recall else np.nan,
        density=round(density, 4),
        enrichment=round(enrichment, 4) if enrichment == enrichment else np.nan,
        n_train_samples=len(selected_set),
        n_covered_incidents=covered if n_incidents else 0,
    )


def scan_index_files(dataset: str,
                     incident_starts: np.ndarray,
                     incidents_by_id: dict) -> pd.DataFrame:
    base_rate = len(incident_starts) / N_TRAIN
    rows = []
    for p in sorted(index_dir(dataset).glob('*.json')):
        if p.name == 'proxy_metrics.json':
            continue
        m = FNAME_RE.match(p.stem)
        if not m:
            continue
        if VARIANT_RE.search(m.group('method')):
            continue
        selected = np.array(json.load(open(p)), dtype=np.int64)
        metrics = coreset_metrics(selected, incident_starts,
                                   incidents_by_id, base_rate)
        rows.append({
            'dataset':  dataset,
            'method':   m.group('method'),
            'distance': m.group('distance'),
            'ratio':    int(m.group('ratio')) / 100,
            'seed':     int(m.group('seed')),
            'base_rate': round(base_rate, 4),
            **metrics,
        })
    return pd.DataFrame(rows)


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Average canonical Euclidean indices over selection seeds.

    Other distances are Phase B ablations and do not supply the main paper
    grid.  Variant filenames were already excluded while scanning.
    """
    main = df[df['distance'] == 'euclidean']
    agg = (main.groupby(['dataset', 'method', 'ratio'])
           .agg(recall=('recall', 'mean'),
                density=('density', 'mean'),
                enrichment=('enrichment', 'mean'),
                n_configs=('seed', 'size'))
           .reset_index())
    return agg.sort_values(['dataset', 'ratio', 'method'])


def correlate_with_mae(method_agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (ratio, dataset), Spearman between incident metrics and MAE.
    Also a pooled-across-datasets row."""
    mae = pd.read_csv(UNIFIED_MAE)
    mae = mae[mae['ratio'] < 1.0]
    merged = mae.merge(method_agg, on=['dataset', 'method', 'ratio'], how='inner')

    rows = []
    for r in sorted(merged['ratio'].unique()):
        for ds_label, sub in (
            [('pooled', merged[merged['ratio'] == r])] +
            [(ds, merged[(merged['ratio'] == r) & (merged['dataset'] == ds)])
             for ds in sorted(merged['dataset'].unique())]
        ):
            if len(sub) < 5:
                continue
            rho_recall, p_recall = spearmanr(sub['recall'], sub['MAE_mean'])
            rho_enrich, p_enrich = spearmanr(sub['enrichment'], sub['MAE_mean'])
            rows.append({
                'ratio': r, 'dataset': ds_label, 'n': len(sub),
                'recall_vs_MAE':     round(rho_recall, 3),
                'recall_p':          round(p_recall, 4),
                'enrichment_vs_MAE': round(rho_enrich, 3),
                'enrichment_p':      round(p_enrich, 4),
            })
    return pd.DataFrame(rows), merged


def main() -> None:
    print(f'Train window: {TRAIN_TIMESTEPS} timesteps, '
          f'N_train = {N_TRAIN} samples')

    all_rows = []
    for ds in DATASETS:
        inc_starts = load_train_incident_starts(ds)
        inc_by_id  = load_incidents_by_id(ds)
        base_rate  = len(inc_starts) / N_TRAIN
        print(f'\n[{ds}] severe incidents in train window: {len(inc_by_id)}, '
              f'incident-touching samples: {len(inc_starts)} '
              f'(base rate {base_rate:.4f})')
        df = scan_index_files(ds, inc_starts, inc_by_id)
        all_rows.append(df)
    per_config = pd.concat(all_rows, ignore_index=True)
    per_config.to_csv(OUTDIR / 'incident_coverage_by_config.csv', index=False)
    print(f'\nPer-config rows: {len(per_config)}')

    by_method = aggregate_by_method(per_config)
    by_method.to_csv(OUTDIR / 'incident_coverage_by_method.csv', index=False)
    print('\n--- incident_coverage_by_method.csv ---')
    print(by_method.to_string(index=False))

    corr, merged = correlate_with_mae(by_method)
    corr.to_csv(OUTDIR / 'incident_coverage_vs_mae_spearman.csv', index=False)
    merged.to_csv(OUTDIR / 'incident_coverage_vs_mae.csv', index=False)
    print('\n--- incident_coverage_vs_mae_spearman.csv (Spearman per ratio) ---')
    print(corr.to_string(index=False))


if __name__ == '__main__':
    main()
