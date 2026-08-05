"""
Parallel severe-incident MAE extraction using multiprocessing.

The sequential version (incident_severe_manual_mae.py) bottlenecks on network
filesystem (QRIS) I/O — each 591 MB npz takes 30-300 s to read. With 1900
files that's infeasible. This version:
  1. Uses multiprocessing (default: 8 workers) to parallelize the slow npz
     loads — the QRIS filesystem supports multiple concurrent reads.
  2. Tightens the phase filter to only the phases that populate the unified
     MAE table we're correlating against.
  3. Resume-safe: reads existing `severe_mae_per_run.csv`, skips already-
     processed run_dirs.

Output is identical schema to the sequential version; both write to the same
CSV, so runs from either script accumulate.
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / 'checkpoints'
OUTDIR = REPO / 'experiments/result/analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / 'severe_mae_per_run.csv'

sys.stdout.reconfigure(line_buffering=True)

N_WORKERS = int(os.environ.get('N_WORKERS', '8'))

# Tight filter: only phases feeding into unified_mae_with_std.csv + refs.
PHASES = {
    'phase_b_deterministic',
    'phase_c_method_comparison',
    'phase_c_extra_ratios',
    'phase_c_dcrnn_no_cl',
    'phase_d_kmedoids_rerun',
    'phase_b_full_data',
    'phase_b_full_data_seed123',
    'coreset',
    'STAEformer',
}
KEEP_SEEDS   = {42}
KEEP_MODELS  = {'AGCRN', 'STID', 'STGCNChebGraphConv'}
KEEP_METHODS = {'k_medoids', 'stride', 'recent', 'graph_cut'}
# Extreme minimum: just r=0.1 (where recall matters most) + r=0.9 (convergence
# check) + r=1.0 (full-data reference).
KEEP_RATIOS  = {0.1, 0.9, 1.0}

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
    inc = pd.read_csv(REPO / f'EDA/severe_incidents_{dataset}.csv')
    t0, t1 = test_start(), test_end()
    slots = {int(s) + 11 for s in inc['input_start_slot']}
    return {s - t0 for s in slots if t0 <= s < t1}


def mae_masked(pred: np.ndarray, target: np.ndarray) -> float:
    m = target != 0.0
    if not m.any():
        return float('nan')
    return float(np.mean(np.abs(pred[m] - target[m])))


def process_run(args: tuple) -> dict | None:
    """Worker: load one npz, compute severe/non-severe MAE, return row."""
    run_dir_str, severe_by_ds = args
    run_dir = Path(run_dir_str)
    cfg = parse_cfg(run_dir / 'cfg.txt')
    if (cfg.get('method') is None or cfg.get('ratio') is None
            or cfg.get('dataset') is None):
        return None
    if cfg.get('seed') is not None and cfg['seed'] not in KEEP_SEEDS:
        return None
    if cfg['method'] not in KEEP_METHODS:
        return None
    if cfg['ratio'] not in KEEP_RATIOS:
        return None
    ds = cfg['dataset']
    severe = severe_by_ds.get(ds)
    if severe is None:
        return None
    # Filter by model via path (model name is parts[1] under CKPT_ROOT).
    parts = run_dir.relative_to(CKPT_ROOT).parts
    model = parts[1] if len(parts) >= 2 else None
    if model not in KEEP_MODELS:
        return None
    npz_path = run_dir / 'test_results.npz'
    if not npz_path.exists():
        return None
    t0 = time.time()
    try:
        npz = np.load(npz_path)
        pred, target = npz['prediction'], npz['target']
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}',
                'run_dir': str(run_dir.relative_to(REPO))}

    n = pred.shape[0]
    mask_severe = np.zeros(n, dtype=bool)
    for i in severe:
        if 0 <= i < n:
            mask_severe[i] = True
    if mask_severe.sum() == 0:
        return None

    mae_sev = mae_masked(pred[mask_severe],  target[mask_severe])
    mae_non = mae_masked(pred[~mask_severe], target[~mask_severe])
    mae_all = mae_masked(pred, target)

    return {
        'phase':    parts[0],
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
        'load_time_s':    round(time.time() - t0, 2),
        'run_dir':        str(run_dir.relative_to(REPO)),
    }


def main() -> None:
    severe_by_ds = {
        ds: load_severe_test_indices(ds)
        for ds in ['SAN_BERNARDINO', 'CONTRA_COSTA']
    }
    for ds, s in severe_by_ds.items():
        print(f'{ds}: {len(s)} severe incidents in test window', flush=True)

    done = set()
    if OUT_CSV.exists():
        try:
            done = set(pd.read_csv(OUT_CSV)['run_dir'])
        except Exception:
            pass
    print(f'Resuming — {len(done)} runs already cached', flush=True)

    all_npz = []
    for phase in sorted(PHASES):
        phase_dir = CKPT_ROOT / phase
        if phase_dir.exists():
            all_npz.extend(sorted(phase_dir.rglob('test_results.npz')))
    tasks = [str(p.parent) for p in all_npz
             if str(p.parent.relative_to(REPO)) not in done]
    print(f'Found {len(all_npz)} npz files; {len(tasks)} to process', flush=True)

    t_start = time.time()
    processed = errors = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_run, (task, severe_by_ds)): task
                   for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as e:
                r = {'error': str(e), 'run_dir': futures[fut]}
            if r is None:
                continue
            if 'error' in r:
                errors += 1
                print(f'  [{i}/{len(tasks)}] ERROR '
                      f'{r["run_dir"][-80:]}: {r["error"]}', flush=True)
                continue
            # Append single row
            df_new = pd.DataFrame([r])
            header = not OUT_CSV.exists()
            df_new.to_csv(OUT_CSV, mode='a', index=False, header=header)
            processed += 1
            if processed <= 5 or processed % 20 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                remaining = (len(tasks) - i) / rate if rate > 0 else float('inf')
                print(f'  [{i}/{len(tasks)}] processed={processed} '
                      f'errors={errors} '
                      f'load_last={r["load_time_s"]:.1f}s '
                      f'rate={rate:.2f}/s '
                      f'eta={remaining/60:.0f}min', flush=True)

    print(f'\nDone: processed={processed} errors={errors} '
          f'total={time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
