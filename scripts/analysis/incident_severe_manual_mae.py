"""
Manually compute severe-incident vs non-severe MAE from saved test predictions.

Why manual
----------
The stored `test_incident_metrics.json` numbers do NOT match manual computation
from `test_results.npz` (cross-check in verify_incident_metrics.py showed
stored says incidents are −0.9 MAE, manual says +0.4–+0.8 MAE). The runner's
per-subset `compute_evaluation_metrics` call appears to mishandle scaling or
masking. This script bypasses that by computing MAE directly from the saved
prediction/target tensors, using our severe-incident definition (z<-2).

Method
------
For each run that has `test_results.npz`:
  1. Parse cfg.txt for (model, dataset, method, ratio, seed, distance)
  2. Compute severe test-sample indices for that dataset:
       slot = input_start_slot + 11     (runner convention)
       rel  = slot - test_start          (relative to test window)
       keep only severe incidents (EDA/severe_incidents_{DATASET}.csv)
  3. mask_severe[i] = True if i in severe set
     MAE_severe     = mean|pred - target| over samples with mask_severe,
                      excluding null_val=0 target entries (BasicTS convention)
     MAE_non_severe = same on the complement
  4. Write one row per run.

Incremental caching
-------------------
Each run takes several seconds — npz I/O dominant, and `checkpoints/` sits on
a networked QRIS filesystem, so I/O is slow and variable. We write rows to a
partial CSV after every successful run; rerunning the script skips already-
processed rows, so a crash loses at most one row's work.

Phase filter
------------
We restrict to phases relevant to the coreset paper:
  phase_b_deterministic, phase_c_method_comparison, phase_c_extra_ratios,
  phase_c_dcrnn_no_cl, phase_d_kmedoids_rerun, phase_b_full_data.
Other phases (adjacency, replay, etc.) are skipped.

Output
------
- experiments/result/analysis/severe_mae_per_run.csv    (one row per run)

Usage
-----
  conda activate cuda && python scripts/analysis/incident_severe_manual_mae.py
"""

from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / 'checkpoints'
OUTDIR = REPO / 'experiments/result/analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / 'severe_mae_per_run.csv'

# Force line-buffered stdout so progress is visible when redirected.
sys.stdout.reconfigure(line_buffering=True)

RELEVANT_PHASES = {
    'phase_b_deterministic',
    'phase_c_method_comparison',
    'phase_c_extra_ratios',
    'phase_c_dcrnn_no_cl',
    'phase_d_kmedoids_rerun',
    'phase_b_full_data',
    'phase_b_full_data_seed123',
    'coreset',   # earlier full-data baseline runs land here
    'STAEformer',  # some full-data STAEformer runs land here
}
# Subsample by seed to keep runtime tractable on the networked FS.
# Using seed=42 across all phases gives us one point per
# (model, dataset, method, ratio) cell — enough for the main-table story.
KEEP_SEEDS = {42}

INPUT_LEN  = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST = (0.6, 0.2, 0.2)
DATA_RANGE_END = 24192

CFG_PAT = {
    'method':   re.compile(r'SELECTION_STRATEGY:\s*(\S+)'),
    'ratio':    re.compile(r'SELECTION_RATIO:\s*([0-9.]+)'),
    'seed':     re.compile(r'\n\s*SEED:\s*(\d+)'),
    'distance': re.compile(r'DISTANCE_TYPE:\s*(\S+)'),
    'dataset':  re.compile(r"dataset_name:\s*xtraffic/(\S+)"),
}


def parse_cfg(path: Path) -> dict:
    try:
        text = path.read_text()
    except Exception:
        return {}
    out = {k: (m.group(1) if (m := pat.search(text)) else None)
           for k, pat in CFG_PAT.items()}
    if out.get('ratio') is not None:
        out['ratio'] = float(out['ratio'])
    if out.get('seed') is not None:
        out['seed'] = int(out['seed'])
    return out


def test_start(total_len: int = DATA_RANGE_END) -> int:
    valid_len = int(total_len * TRAIN_VAL_TEST[1])
    test_len  = int(total_len * TRAIN_VAL_TEST[2])
    train_len = total_len - valid_len - test_len
    return train_len + valid_len - (INPUT_LEN - 1)


def test_end(total_len: int = DATA_RANGE_END) -> int:
    return test_start(total_len) + int(total_len * TRAIN_VAL_TEST[2])


def load_severe_test_indices(dataset: str) -> set:
    """Relative test-window indices of severe incidents (z<-2)."""
    inc = pd.read_csv(REPO / f'EDA/severe_incidents_{dataset}.csv')
    t0, t1 = test_start(), test_end()
    slots = {int(s) + 11 for s in inc['input_start_slot']}
    return {s - t0 for s in slots if t0 <= s < t1}


def mae_null_masked(pred: np.ndarray, target: np.ndarray,
                    null_val: float = 0.0) -> float:
    mask = target != null_val
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def evaluate_run(run_dir: Path, severe_idx_by_ds: dict) -> dict | None:
    cfg = parse_cfg(run_dir / 'cfg.txt')
    if cfg.get('method') is None or cfg.get('ratio') is None \
       or cfg.get('dataset') is None:
        return None
    if cfg.get('seed') is not None and cfg['seed'] not in KEEP_SEEDS:
        return None
    ds = cfg['dataset']
    severe_idx = severe_idx_by_ds.get(ds)
    if severe_idx is None:
        return None

    npz_path = run_dir / 'test_results.npz'
    if not npz_path.exists():
        return None
    try:
        npz = np.load(npz_path)
        pred, target = npz['prediction'], npz['target']
    except Exception as e:
        return {'error': str(e)}

    n = pred.shape[0]
    if n == 0:
        return None
    mask_severe = np.zeros(n, dtype=bool)
    for i in severe_idx:
        if 0 <= i < n:
            mask_severe[i] = True
    if mask_severe.sum() == 0:
        return None

    mae_sev = mae_null_masked(pred[mask_severe],  target[mask_severe])
    mae_non = mae_null_masked(pred[~mask_severe], target[~mask_severe])
    mae_all = mae_null_masked(pred, target)

    parts = run_dir.relative_to(CKPT_ROOT).parts
    phase = parts[0]
    model = parts[1] if len(parts) >= 2 else None

    return {
        'phase':    phase,
        'model':    model,
        'dataset':  ds,
        'method':   cfg['method'],
        'ratio':    cfg['ratio'],
        'seed':     cfg.get('seed'),
        'distance': cfg.get('distance'),
        'n_test':   int(n),
        'n_severe': int(mask_severe.sum()),
        'MAE_overall':    round(mae_all, 4),
        'MAE_severe':     round(mae_sev, 4),
        'MAE_non_severe': round(mae_non, 4),
        'run_dir': str(run_dir.relative_to(REPO)),
    }


def main() -> None:
    severe_idx_by_ds = {
        ds: load_severe_test_indices(ds)
        for ds in ['SAN_BERNARDINO', 'CONTRA_COSTA']
    }
    for ds, s in severe_idx_by_ds.items():
        print(f'{ds}: {len(s)} severe incidents in test window')

    # Resume-friendly: read existing CSV, skip already-done run_dirs.
    done = set()
    if OUT_CSV.exists():
        try:
            cached = pd.read_csv(OUT_CSV)
            done = set(cached['run_dir'])
            print(f'Resuming — {len(done)} runs already in {OUT_CSV.name}')
        except Exception:
            pass

    # Gather npz files from only the relevant phases (avoids walking network
    # directories for adjacency/replay/etc. which we don't need here).
    all_npz = []
    for phase in sorted(RELEVANT_PHASES):
        phase_dir = CKPT_ROOT / phase
        if not phase_dir.exists():
            continue
        all_npz.extend(sorted(phase_dir.rglob('test_results.npz')))
    print(f'Found {len(all_npz)} test_results.npz files across '
          f'{len(RELEVANT_PHASES)} phases', flush=True)

    rows = []
    t_start = time.time()
    skipped = errors = 0
    new_count = 0

    for i, npz_path in enumerate(all_npz):
        run_dir = npz_path.parent
        rel = str(run_dir.relative_to(REPO))
        if rel in done:
            skipped += 1
            continue
        t0 = time.time()
        try:
            r = evaluate_run(run_dir, severe_idx_by_ds)
        except Exception as e:
            r = {'error': f'{type(e).__name__}: {e}'}
        if r is None:
            continue
        if 'error' in r:
            errors += 1
            print(f'  [{i+1}/{len(all_npz)}] ERROR {rel[:90]}: {r["error"]}',
                  flush=True)
            continue
        # Flush every row (single-row append) — so the CSV reflects truth even
        # if the job is interrupted. Networked FS writes are fast for CSV
        # compared to npz reads, so overhead is negligible.
        _flush([r], OUT_CSV)
        new_count += 1
        if new_count <= 5 or new_count % 25 == 0:
            elapsed = time.time() - t_start
            print(f'  [{i+1}/{len(all_npz)}] processed={new_count} '
                  f'skipped={skipped} errors={errors} '
                  f'last={time.time()-t0:.1f}s '
                  f'avg={elapsed/max(new_count,1):.2f}s/run '
                  f'total={elapsed:.0f}s',
                  flush=True)

    print(f'\nDone: new rows={new_count}, total skipped={skipped}, '
          f'errors={errors}, total time={time.time()-t_start:.0f}s', flush=True)


def _flush(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if out_csv.exists():
        df_new.to_csv(out_csv, mode='a', index=False, header=False)
    else:
        df_new.to_csv(out_csv, index=False)


if __name__ == '__main__':
    main()
