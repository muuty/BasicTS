#!/usr/bin/env python3
"""State-conditional forecasting MAE versus FD support difficulty.

This is the empirical bridge between the detector-resolved fundamental-diagram
support audit (``fd_sensor_resolved.py``) and the models' realised test error.
The support audit measures how hard each detector-local traffic-state
transition is to *replace* from a reduced time support.  It does not, by
itself, show that this difficulty translates into worse forecasting.

This script closes that gap without any retraining or re-inference.  It reuses
the archived test predictions (``test_results.npz``) that every phase-C run
already wrote, classifies each TEST window into the same detector-resolved
transition (free/breakdown/recovery/congested) used by the audit, and computes
masked forecasting MAE per transition.  Comparing a reduced-data run against
the full-data run isolates the state-specific degradation caused by data
reduction.

Design decisions:
  * The FD screen and per-detector critical points are read from
    ``fd_sensor_resolved_fits.csv`` (the paper's source of truth), never
    recomputed, so the retained detector set and thresholds match exactly.
  * State classification runs on the chronological TEST split, using per
    detector critical occupancy / speed-at-capacity fitted on the TRAIN split
    (no leakage).
  * Test-window indexing mirrors ``basicts.data.simple_tsf_dataset`` exactly:
    with ``data_range=(0, 24192)`` and a 0.6/0.2/0.2 split the test region is
    ``data[19354:24192]`` and window ``i`` predicts absolute steps
    ``[19354+i+12, 19354+i+24)``.

The output is a descriptive, state-level association at a fixed training seed.
It is not a causal or seed-uncertainty statement, and (like the audit) it is
not a certified model-risk bound.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
FITS_CSV = OUT / "fd_sensor_resolved_fits.csv"
PHASE_C = REPO / "checkpoints" / "phase_c_method_comparison"

# Split geometry, identical to fd_sensor_resolved.py and the dataset loader.
RANGE_END = 24_192
VALID_STEPS = int(RANGE_END * 0.2)
TEST_STEPS = int(RANGE_END * 0.2)
TRAIN_STEPS = RANGE_END - VALID_STEPS - TEST_STEPS
TEST_OFFSET = TRAIN_STEPS + VALID_STEPS  # first absolute step of the test region
T_IN = T_OUT = 12
NULL_VAL = 0.0

DATASETS = ("SAN_BERNARDINO", "CONTRA_COSTA")
# Directory name under phase_c for each backbone.
BACKBONE_DIRS = {
    "STGCN": "STGCNChebGraphConv",
    "DCRNN": "DCRNN",
    "STAEformer": "STAEformer",
    "STID": "STID",
    "AGCRN": "AGCRN",
}
STATE_NAMES = {0: "free_to_free", 1: "breakdown", 2: "recovery", 3: "congested_to_congested"}
# Full-data reference is a 100%-selection run; reduced runs are compared to it.
FULL_KEY = ("random", 1.0)


def parse_cfg(cfg_path: Path) -> tuple[str, float]:
    text = cfg_path.read_text()
    strat = re.search(r"SELECTION_STRATEGY:\s*(\S+)", text)
    ratio = re.search(r"SELECTION_RATIO:\s*([\d.]+)", text)
    strategy = strat.group(1) if strat else ("full" if "CORESET:" not in text else "unknown")
    return strategy, (float(ratio.group(1)) if ratio else 1.0)


def leaf_map(backbone_dir: str, dataset: str) -> dict[tuple[str, float], list[Path]]:
    base = PHASE_C / backbone_dir / "xtraffic" / f"{dataset}_50_12_12"
    groups: dict[tuple[str, float], list[Path]] = {}
    for npz in sorted(base.glob("*/*/test_results.npz")):
        cfg = npz.parent / "cfg.txt"
        if cfg.exists():
            groups.setdefault(parse_cfg(cfg), []).append(npz)
    return groups


def horizon_transition(observation_state: np.ndarray, min_valid: int, min_agreement: float) -> np.ndarray:
    """Aggregate per-step states into per-window transition codes.

    ``observation_state`` is ``(T, S)`` in {-1 unresolved, 0 free, 1 congested}
    over a contiguous region.  Returns ``(W, S)`` transition codes for windows
    where history is ``[i, i+T_IN)`` and future is ``[i+T_IN, i+2 T_IN)``.
    """
    T, S = observation_state.shape
    free = (observation_state == 0).astype(np.int32)
    cong = (observation_state == 1).astype(np.int32)
    free_prefix = np.vstack([np.zeros((1, S), np.int32), free.cumsum(0)])
    cong_prefix = np.vstack([np.zeros((1, S), np.int32), cong.cumsum(0)])

    def window_state(starts: np.ndarray) -> np.ndarray:
        stop = starts + T_IN
        n_free = free_prefix[stop] - free_prefix[starts]
        n_cong = cong_prefix[stop] - cong_prefix[starts]
        n_valid = n_free + n_cong
        state = np.full(n_free.shape, -1, np.int8)
        valid = n_valid >= min_valid
        state[valid & (n_free >= min_agreement * n_valid)] = 0
        state[valid & (n_cong >= min_agreement * n_valid)] = 1
        return state

    starts = np.arange(T - 2 * T_IN + 1)
    history = window_state(starts)
    future = window_state(starts + T_IN)
    code = np.full(history.shape, -1, np.int8)
    for h, f, c in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        code[(history == h) & (future == f)] = c
    return code


def test_transitions(dataset: str, accepted: pd.DataFrame, min_valid: int, min_agreement: float):
    """Per-detector transition code for every test window, plus sensor ids."""
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(
        REPO / "datasets" / dataset / "data.dat",
        dtype=np.float32, mode="r", shape=tuple(desc["shape"]),
    )
    sensors = accepted["sensor"].to_numpy(int)
    occ = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 1], np.float32)
    speed = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 2], np.float32)
    oc = accepted["critical_occupancy"].to_numpy(float)[None, :]
    vc = accepted["speed_at_capacity"].to_numpy(float)[None, :]
    obs = np.full(occ.shape, -1, np.int8)
    obs[(occ < oc) & (speed > vc)] = 0
    obs[(occ > oc) & (speed < vc)] = 1
    return sensors, horizon_transition(obs, min_valid, min_agreement)


def window_detector_mae(npz_paths: list[Path], sensors: np.ndarray, max_seeds: int) -> np.ndarray:
    """Masked per-(window, detector) MAE, averaged over up to ``max_seeds`` runs."""
    per_seed = []
    for path in npz_paths[:max_seeds]:
        data = np.load(path)
        pred = np.asarray(data["prediction"][:, :, sensors, 0])  # (W, T_OUT, S)
        target = np.asarray(data["target"][:, :, sensors, 0])
        mask = target != NULL_VAL
        numer = (np.abs(pred - target) * mask).sum(axis=1)
        denom = mask.sum(axis=1)
        per_seed.append(np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0))
    return np.nanmean(np.stack(per_seed), axis=0)  # (W, S)


def detector_state_rows(wd_mae: np.ndarray, code: np.ndarray, sensors: np.ndarray,
                        dataset: str, backbone: str, method: str, ratio: float) -> list[dict]:
    rows = []
    for local_s in range(code.shape[1]):
        column = wd_mae[:, local_s]
        for state_code, state_name in STATE_NAMES.items():
            vals = column[code[:, local_s] == state_code]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                rows.append({
                    "dataset": dataset, "backbone": backbone, "method": method,
                    "ratio": ratio, "sensor": int(sensors[local_s]),
                    "traffic_state_transition": state_name,
                    "mae": float(vals.mean()), "n_windows": int(len(vals)),
                })
    return rows


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS_CSV)
    min_valid = int(math.ceil(T_IN * args.min_valid_fraction))
    cell_keys = [FULL_KEY] + [(m, args.ratio) for m in args.methods]

    detector_rows: list[dict] = []
    for dataset in args.datasets:
        accepted = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor").reset_index(drop=True)
        sensors, code = test_transitions(dataset, accepted, min_valid, args.min_agreement)
        atoms = {name: int((code == c).sum()) for c, name in STATE_NAMES.items()}
        print(f"[{dataset}] detectors={len(sensors)} test-window atoms={atoms}", flush=True)
        for backbone, backbone_dir in BACKBONE_DIRS.items():
            if backbone not in args.backbones:
                continue
            leaves = leaf_map(backbone_dir, dataset)
            for method, ratio in cell_keys:
                paths = leaves.get((method, ratio))
                if not paths:
                    print(f"  {backbone:11s} [skip] no npz for {method} r={ratio}", flush=True)
                    continue
                label = "full" if (method, ratio) == FULL_KEY else method
                wd_mae = window_detector_mae(paths, sensors, args.max_seeds)
                detector_rows.extend(detector_state_rows(
                    wd_mae, code, sensors, dataset, backbone, label, ratio))
                print(f"  {backbone:11s} {label:10s} r={ratio}: "
                      f"{min(len(paths), args.max_seeds)}/{len(paths)} seeds", flush=True)

    by_detector = pd.DataFrame(detector_rows)
    by_detector.to_csv(OUT / "fd_state_conditional_mae_by_detector.csv", index=False)

    # Pooled summary and degradation versus the full-data run per backbone/dataset.
    summary = (by_detector.groupby(
        ["dataset", "backbone", "method", "ratio", "traffic_state_transition"], as_index=False)
        .agg(mae_detector_mean=("mae", "mean"), n_detectors=("sensor", "nunique"),
             n_windows=("n_windows", "sum")))
    full = summary[summary.method == "full"][
        ["dataset", "backbone", "traffic_state_transition", "mae_detector_mean"]
    ].rename(columns={"mae_detector_mean": "mae_full"})
    summary = summary.merge(full, on=["dataset", "backbone", "traffic_state_transition"], how="left")
    summary["degradation_vs_full"] = summary["mae_detector_mean"] - summary["mae_full"]
    summary.to_csv(OUT / "fd_state_conditional_mae_summary.csv", index=False)

    report_robustness(summary, args.methods)
    print("\nwrote fd_state_conditional_mae_{by_detector,summary}.csv", flush=True)


def report_robustness(summary: pd.DataFrame, methods: tuple[str, ...]) -> None:
    """Print the headline: state-specific degradation relative to free flow."""
    for method in methods:
        sub = summary[summary.method == method]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=["dataset", "backbone"],
                                columns="traffic_state_transition", values="degradation_vs_full")
        free = pivot["free_to_free"]
        print(f"\n=== {method} r: degradation minus free-flow degradation (MAE units) ===")
        print(f"{'dataset':16s}{'backbone':12s}{'breakdown':>11s}{'recovery':>10s}{'congested':>11s}")
        for (dataset, backbone), row in pivot.iterrows():
            bd = row.get("breakdown", np.nan) - free[(dataset, backbone)]
            rec = row.get("recovery", np.nan) - free[(dataset, backbone)]
            cong = row.get("congested_to_congested", np.nan) - free[(dataset, backbone)]
            print(f"{dataset:16s}{backbone:12s}{bd:11.2f}{rec:10.2f}{cong:11.2f}")
        for state in ("breakdown", "recovery", "congested_to_congested"):
            diff = pivot[state] - free
            print(f"  {state:24s} minus free: median={diff.median():+.2f}  "
                  f">0 in {(diff > 0).sum()}/{diff.notna().sum()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONE_DIRS), default=list(BACKBONE_DIRS))
    parser.add_argument("--methods", nargs="+", default=["k_medoids", "random"],
                        help="Reduced-data selectors to compare against full data.")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--max-seeds", type=int, default=3,
                        help="Selection-seed runs to average per cell (cold I/O bound).")
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    main()
