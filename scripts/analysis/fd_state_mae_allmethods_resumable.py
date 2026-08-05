#!/usr/bin/env python3
"""Per-(detector, state) MAE for all six selectors at r=0.3, resumable.

Writes one row per (dataset, backbone, method, sensor, state) by reading each
model's test_results.npz. Each (dataset, backbone, method) cell is appended to
the output CSV as soon as it is computed, and cells already present are skipped
on restart, so the job survives interruption and resumes where it stopped.

Reuses the labelling and MAE machinery in fd_state_conditional_mae.py unchanged.
Output: experiments/result/analysis/fd_state_mae_allmethods_by_detector.csv
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from fd_state_conditional_mae import (  # noqa: E402
    BACKBONE_DIRS, DATASETS, FITS_CSV, FULL_KEY, OUT, STATE_NAMES, T_IN,
    detector_state_rows, leaf_map, test_transitions, window_detector_mae,
)

OUTFILE = OUT / "fd_state_mae_allmethods_by_detector.csv"
METHODS = ["k_medoids", "k_center", "graph_cut", "random", "recent", "stride"]
RATIO = 0.3
MAX_SEEDS = 3
MIN_VALID_FRACTION = 2 / 3
MIN_AGREEMENT = 0.75


def done_cells() -> set:
    if OUTFILE.exists():
        df = pd.read_csv(OUTFILE, usecols=["dataset", "backbone", "method"])
        return set(map(tuple, df.drop_duplicates().to_numpy()))
    return set()


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUTFILE, mode="a", header=not OUTFILE.exists(), index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS_CSV)
    min_valid = int(math.ceil(T_IN * MIN_VALID_FRACTION))
    done = done_cells()
    print(f"resuming: {len(done)} cells already done", flush=True)
    cell_keys = [FULL_KEY] + [(m, RATIO) for m in METHODS]

    for dataset in DATASETS:
        accepted = (fits[(fits.dataset == dataset) & fits.identifiable]
                    .sort_values("sensor").reset_index(drop=True))
        sensors, code = test_transitions(dataset, accepted, min_valid, MIN_AGREEMENT)
        atoms = {name: int((code == c).sum()) for c, name in STATE_NAMES.items()}
        print(f"[{dataset}] detectors={len(sensors)} atoms={atoms}", flush=True)
        for backbone, backbone_dir in BACKBONE_DIRS.items():
            leaves = leaf_map(backbone_dir, dataset)
            for method, ratio in cell_keys:
                label = "full" if (method, ratio) == FULL_KEY else method
                if (dataset, backbone, label) in done:
                    print(f"  {backbone:11s} {label:10s}: skip (done)", flush=True)
                    continue
                paths = leaves.get((method, ratio))
                if not paths:
                    print(f"  {backbone:11s} {label:10s}: no npz", flush=True)
                    continue
                wd_mae = window_detector_mae(paths, sensors, MAX_SEEDS)
                rows = detector_state_rows(
                    wd_mae, code, sensors, dataset, backbone, label, ratio)
                append_rows(rows)
                print(f"  {backbone:11s} {label:10s}: WROTE {len(rows)} rows "
                      f"({min(len(paths), MAX_SEEDS)}/{len(paths)} seeds)", flush=True)
    print("ALL_CELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
