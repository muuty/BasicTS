"""
Cross-check the stored `test_incident_metrics.json` numbers.

Background
----------
The `IncidentAwareRunner` defines "incident" as ANY row in
`incident_metadata_2023.csv` (25015 rows for SB) — mostly hazards and minor
events, NOT the 335 severe flow-drop incidents we identified by z-score < -2.

When I reported "incident MAE is LOWER than non-incident MAE", that was using
the runner's definition. The user's intuition ("elsewhere incidents had higher
MAE") may be based on SEVERE incidents only. This script re-computes MAE on
the test set with TWO definitions side-by-side:

  (a) runner-default: all 25015 `incident_metadata_2023.csv` rows
  (b) severe-only:   the 335 z<-2 rows from EDA/severe_incidents_{DS}.csv

and compares both to non-incident MAE. We do this for a few sampled runs and
confirm whether the "incidents are easier" result holds only under (a) or
also under (b).

Outputs (printed)
-----------------
For each sampled run:
  - Stored non_incident_MAE, all_incident_MAE
  - Manual (a) non_incident_MAE, all_incident_MAE — should match stored
  - Manual (b) non_incident_MAE, severe_incident_MAE — the cross-check

Usage
-----
  conda activate cuda && python scripts/analysis/verify_incident_metrics.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / 'checkpoints'

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST = (0.6, 0.2, 0.2)
DATA_RANGE_END = 24192  # 3-month window


def test_start_end(total_len: int = DATA_RANGE_END) -> tuple[int, int]:
    """Replicate `_get_test_data_start_index` from incident_aware_runner.
    Uses: valid_len = floor(total*0.2), test_len = floor(total*0.2),
          train_len = total - valid - test
          test_start = train_len + valid_len - (INPUT_LEN - 1)   (overlap=True)
          test_end   = test_start + test_len"""
    valid_len = int(total_len * TRAIN_VAL_TEST[1])
    test_len  = int(total_len * TRAIN_VAL_TEST[2])
    train_len = total_len - valid_len - test_len
    offset = INPUT_LEN - 1
    test_start = train_len + valid_len - offset
    test_end   = test_start + test_len
    return test_start, test_end


def mae_on_mask(pred: np.ndarray, target: np.ndarray,
                null_val: float = 0.0) -> float:
    """MAE with null-value masking. BasicTS convention: ignore target==0."""
    mask = target != null_val
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def evaluate_run(run_dir: Path, dataset: str) -> dict:
    npz = np.load(run_dir / 'test_results.npz')
    pred   = npz['prediction']
    target = npz['target']
    n_test = pred.shape[0]

    test_start, test_end = test_start_end()

    # (a) runner default — all incidents from incident_metadata_2023.csv
    inc_all = pd.read_csv(REPO / f'datasets/xtraffic/{dataset}/incident_metadata_2023.csv')
    slots_a = {int(s) + 11 for s in inc_all['input_start_slot']}  # runner formula
    rel_a = {s - test_start for s in slots_a if test_start <= s < test_end}

    # (b) severe-only
    inc_sev = pd.read_csv(REPO / f'EDA/severe_incidents_{dataset}.csv')
    slots_b = {int(s) + 11 for s in inc_sev['input_start_slot']}
    rel_b = {s - test_start for s in slots_b if test_start <= s < test_end}

    # Per-sample indices — runner uses `indices` from dataloader which match
    # test_results.npz row order (sequential 0..n_test-1)
    all_idx = np.arange(n_test)

    def stratified(rel_set: set) -> tuple[float, float, int, int]:
        mask_inc = np.array([i in rel_set for i in all_idx], dtype=bool)
        if mask_inc.sum() == 0:
            return float('nan'), mae_on_mask(pred, target), 0, int(mask_inc.size)
        mae_inc = mae_on_mask(pred[mask_inc], target[mask_inc])
        mae_non = mae_on_mask(pred[~mask_inc], target[~mask_inc])
        return mae_inc, mae_non, int(mask_inc.sum()), int((~mask_inc).sum())

    a_inc, a_non, a_n_inc, a_n_non = stratified(rel_a)
    b_inc, b_non, b_n_inc, b_n_non = stratified(rel_b)

    stored = json.load(open(run_dir / 'test_incident_metrics.json'))
    return {
        'test_start': test_start, 'test_end': test_end, 'n_test': n_test,
        # runner-definition numbers
        'stored_all_incident_MAE':   stored.get('all_incident_overall', {}).get('MAE'),
        'stored_non_incident_MAE':   stored.get('non_incident_overall', {}).get('MAE'),
        'manual_a_incident_MAE':     a_inc,
        'manual_a_non_incident_MAE': a_non,
        'a_n_incident':              a_n_inc,
        'a_n_non_incident':          a_n_non,
        # severe-only numbers (our z<-2 definition)
        'manual_b_severe_MAE':       b_inc,
        'manual_b_non_severe_MAE':   b_non,
        'b_n_severe':                b_n_inc,
        'b_n_non_severe':            b_n_non,
    }


def main() -> None:
    # Pick a handful of diverse runs: different methods × ratios
    # Use phase_c_method_comparison as a source (has many configs)
    root = CKPT_ROOT / 'phase_c_method_comparison' / 'AGCRN' / 'xtraffic' / 'SAN_BERNARDINO_50_12_12' / '3'
    picks = []
    for run_hash_dir in sorted(root.iterdir())[:8]:
        if not (run_hash_dir / 'test_results.npz').exists():
            continue
        cfg = (run_hash_dir / 'cfg.txt').read_text()
        import re
        method = re.search(r'SELECTION_STRATEGY:\s*(\S+)', cfg)
        ratio  = re.search(r'SELECTION_RATIO:\s*([0-9.]+)', cfg)
        seed   = re.search(r'\n\s*SEED:\s*(\d+)', cfg)
        if method and ratio:
            picks.append((run_hash_dir, method.group(1),
                          float(ratio.group(1)),
                          int(seed.group(1)) if seed else -1))

    print(f'Cross-checking {len(picks)} AGCRN/SAN_BERNARDINO runs:\n')
    for d, method, ratio, seed in picks:
        print(f'--- {method} r={ratio} seed={seed} ---')
        try:
            r = evaluate_run(d, 'SAN_BERNARDINO')
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
        print(f'  test window samples: {r["n_test"]}  '
              f'(a) incident={r["a_n_incident"]}  (b) severe={r["b_n_severe"]}')
        print(f'  stored:   non_inc={r["stored_non_incident_MAE"]:.4f}  '
              f'all_inc={r["stored_all_incident_MAE"]:.4f}  '
              f'Δ={r["stored_all_incident_MAE"]-r["stored_non_incident_MAE"]:+.4f}')
        print(f'  (a) all:  non_inc={r["manual_a_non_incident_MAE"]:.4f}  '
              f'all_inc={r["manual_a_incident_MAE"]:.4f}  '
              f'Δ={r["manual_a_incident_MAE"]-r["manual_a_non_incident_MAE"]:+.4f}')
        print(f'  (b) sev:  non_sev={r["manual_b_non_severe_MAE"]:.4f}  '
              f'severe ={r["manual_b_severe_MAE"]:.4f}  '
              f'Δ={r["manual_b_severe_MAE"]-r["manual_b_non_severe_MAE"]:+.4f}')
        print()


if __name__ == '__main__':
    main()
