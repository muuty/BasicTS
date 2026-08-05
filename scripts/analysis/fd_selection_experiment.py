#!/usr/bin/env python3
"""Breakdown-coverage frontier: does raising breakdown exposure recover its loss?

The detector-resolved audit measures where reduction costs accuracy; it does not show
whether a selection objective can buy that accuracy back.  This script joins the two
sides for every r=0.1 objective that has been trained:

  selection side   share of each state's atoms the objective retains, and the calendar
                   total variation of the retained set, both computed on the training
                   split from the two-branch screen (no test-period information);
  outcome side     state-conditional test MAE, its degradation against the full-data
                   run, and the signed error, read from the archived test predictions.

The signed error is reported because the failure at the critical point is a shift and
not added dispersion: under an L1 objective the population-optimal point forecast is the
conditional median, which in a two-branch mixture moves between branches as the branch
weight crosses one half, so a training measure that over-represents the congested branch
biases the forecast toward it.

Reads every checkpoint root in ROOTS, so baselines, the group-balanced runs and the
frontier runs are pooled into one table.

Writes experiments/result/analysis/fd_selection_experiment_{by_detector,summary}.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import fd_state_conditional_mae as sc  # noqa: E402
import fd_spatial_synchronisation as sync  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"
ROOTS = ("phase_c_method_comparison", "phase_c_extra_ratios", "phase_d_kmedoids_rerun",
         "fd_stratified_r010", "fd_breakdown_coverage_r010", "fd_two_state_r010", "fd_split_r030", "fd_frontier_r010")
STATES = {0: "free", 1: "breakdown", 2: "recovery", 3: "sustained"}
STATE_ORDER = ["free", "breakdown", "recovery", "sustained"]
FULL_KEY = ("random", 1.0)


# --------------------------------------------------------------------------
# selection side
# --------------------------------------------------------------------------
def calendar_tv(idx: np.ndarray, n_windows: int) -> float:
    """Total variation between the retained and full (hour, day-of-week) densities."""
    def counts(a):
        c = np.zeros((7, 24))
        np.add.at(c, ((a // 288) % 7, (a % 288) // 12), 1)
        return c / c.sum()
    return float(0.5 * np.abs(counts(idx) - counts(np.arange(n_windows))).sum())


_LABELS: dict[str, np.ndarray] = {}


def training_labels(dataset: str, screen_args) -> np.ndarray:
    """Training-split state codes, fitted once per dataset."""
    if dataset not in _LABELS:
        _, transition, _ = sync.label_matrix(dataset, screen_args)
        _LABELS[dataset] = transition
    return _LABELS[dataset]


def selection_profile(dataset: str, methods: list[str], ratio: float, seed: int,
                      screen_args) -> pd.DataFrame:
    """Retained share of each state's atoms, plus calendar TV, per objective."""
    transition = training_labels(dataset, screen_args)
    n_windows = transition.shape[0]
    counts = {c: (transition == c).sum(axis=1).astype(float) for c in STATES}
    totals = {c: counts[c].sum() for c in STATES}
    ratio_str = f"{ratio:.2f}".replace(".", "")

    rows = []
    for method in methods:
        path = REPO / "coreset_indices" / dataset / f"{method}_euclidean_{ratio_str}_seed{seed}.json"
        if not path.exists():
            continue
        idx = np.asarray(json.loads(path.read_text()), dtype=int)
        idx = idx[idx < n_windows]
        row = {"dataset": dataset, "method": method, "ratio": ratio,
               "calendar_tv": calendar_tv(idx, n_windows)}
        for code, name in STATES.items():
            row[f"coverage_{name}"] = float(counts[code][idx].sum() / totals[code])
        rows.append(row)

    k = int(round(ratio * n_windows))
    oracle = np.argsort(-counts[1], kind="stable")[:k]
    row = {"dataset": dataset, "method": "oracle_breakdown_count", "ratio": ratio,
           "calendar_tv": calendar_tv(oracle, n_windows)}
    for code, name in STATES.items():
        row[f"coverage_{name}"] = float(counts[code][oracle].sum() / totals[code])
    rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# outcome side
# --------------------------------------------------------------------------
def runs_for(backbone_dir: str, dataset: str) -> dict[tuple[str, float], list[Path]]:
    groups: dict[tuple[str, float], list[Path]] = {}
    for root in ROOTS:
        base = REPO / "checkpoints" / root / backbone_dir / "xtraffic" / f"{dataset}_50_12_12"
        if not base.exists():
            continue
        for npz in sorted(base.glob("*/*/test_results.npz")):
            cfg = npz.parent / "cfg.txt"
            if cfg.exists():
                groups.setdefault(sc.parse_cfg(cfg), []).append(npz)
    return groups


CACHE = OUT / "_state_rows_cache"


def state_rows(npz: Path, sensors: np.ndarray, code: np.ndarray, keys: dict) -> list[dict]:
    """Per-detector state-conditional MAE and signed error for one run.

    The archived predictions are 600 MB apiece and never change once written, so the
    per-detector reduction is cached on the npz path and its modification time. A
    re-run that adds one objective then costs one file read, not five hundred.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{abs(hash(str(npz))):016x}_{int(npz.stat().st_mtime)}_{len(sensors)}.npy"
    cached = CACHE / tag
    if cached.exists():
        per = np.load(cached)
    else:
        # Only the screened detectors are needed; memory-map so the rest is never read.
        data = np.load(npz, mmap_mode="r")
        pred = np.asarray(data["prediction"][:, :, sensors, 0], dtype=np.float32)
        target = np.asarray(data["target"][:, :, sensors, 0], dtype=np.float32)
        mask = target != sc.NULL_VAL
        err = pred - target
        # per[state, detector] = (n, sum|err|, sum err), vectorised over windows
        per = np.zeros((len(STATES), code.shape[1], 3), dtype=np.float64)
        for state_code in STATES:
            sel = (code == state_code)[:, None, :] & mask
            per[state_code, :, 0] = sel.sum(axis=(0, 1))
            per[state_code, :, 1] = np.where(sel, np.abs(err), 0.0).sum(axis=(0, 1))
            per[state_code, :, 2] = np.where(sel, err, 0.0).sum(axis=(0, 1))
        np.save(cached, per)

    rows = []
    for state_code, state_name in STATES.items():
        for local_s in range(per.shape[1]):
            n = per[state_code, local_s, 0]
            if n == 0:
                continue
            rows.append({**keys, "sensor": int(sensors[local_s]),
                         "state": state_name, "n_obs": int(n),
                         "mae": float(per[state_code, local_s, 1] / n),
                         "bias": float(per[state_code, local_s, 2] / n)})
    return rows


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    screen_args = types.SimpleNamespace(
        n_bins=40, min_branch_bins=6, min_free_rise=0.10, min_congested_drop=0.05,
        min_fit_improvement=0.10, min_valid_fraction=2 / 3, min_agreement=0.75)
    fits = pd.read_csv(sc.FITS_CSV)
    min_valid = int(math.ceil(sc.T_IN * (2 / 3)))

    selection, detector_rows = [], []
    for dataset in args.datasets:
        for r in args.ratios:
            selection.append(selection_profile(dataset, args.methods, r, args.seed, screen_args))
        accepted = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor").reset_index(drop=True)
        sensors, code = sc.test_transitions(dataset, accepted, min_valid, 0.75)
        print(f"[{dataset}] detectors={len(sensors)}", flush=True)
        for backbone, backbone_dir in sc.BACKBONE_DIRS.items():
            groups = runs_for(backbone_dir, dataset)
            cell_keys = [FULL_KEY] + [(m, r) for r in args.ratios for m in args.methods]
            for key in cell_keys:
                paths = groups.get(key)
                if not paths:
                    continue
                label = "full" if key == FULL_KEY else key[0]
                keys = {"dataset": dataset, "backbone": backbone, "method": label, "ratio": key[1]}
                detector_rows.extend(state_rows(paths[0], sensors, code, keys))
                print(f"  {backbone:11s} {label:22s} r={key[1]}", flush=True)

    by_detector = pd.DataFrame(detector_rows)
    by_detector.to_csv(OUT / "fd_selection_experiment_by_detector.csv", index=False)

    summary = (by_detector.groupby(["dataset", "backbone", "method", "ratio", "state"], as_index=False)
               .agg(mae=("mae", "mean"), bias=("bias", "mean"), n_detectors=("sensor", "nunique")))
    full = (summary[summary.method == "full"][["dataset", "backbone", "state", "mae", "bias"]]
            .rename(columns={"mae": "mae_full", "bias": "bias_full"}))
    summary = summary.merge(full, on=["dataset", "backbone", "state"], how="left")
    summary["degradation"] = summary["mae"] - summary["mae_full"]
    selection_df = pd.concat(selection, ignore_index=True).drop_duplicates(["dataset", "method", "ratio"])
    summary = summary.merge(selection_df, on=["dataset", "method", "ratio"], how="left")
    summary.to_csv(OUT / "fd_selection_experiment_summary.csv", index=False)

    report(summary)
    print("\nwrote fd_selection_experiment_{by_detector,summary}.csv", flush=True)


def report(summary: pd.DataFrame) -> None:
    """The frontier: breakdown coverage against breakdown degradation and bias."""
    red = summary[summary.method != "full"]
    for dataset, sub in red.groupby("dataset"):
        print(f"\n=== {dataset}: breakdown-coverage frontier (architectures averaged) ===", flush=True)
        agg = (sub.groupby(["method", "state"], as_index=False)
               .agg(degradation=("degradation", "mean"), bias=("bias", "mean"),
                    coverage_breakdown=("coverage_breakdown", "first"),
                    coverage_free=("coverage_free", "first"),
                    coverage_sustained=("coverage_sustained", "first"),
                    calendar_tv=("calendar_tv", "first"), n=("backbone", "nunique")))
        bd = agg[agg.state == "breakdown"].sort_values("coverage_breakdown")
        head = (f"{'method':22s}{'cov_bd':>8s}{'cov_free':>9s}{'cov_sust':>9s}{'TV':>6s}"
                f"{'deg_bd':>8s}{'bias_bd':>9s}" + "".join(f"{'deg_' + s:>10s}" for s in STATE_ORDER) + f"{'n':>4s}")
        print(head, flush=True)
        for _, r in bd.iterrows():
            degs = []
            for s in STATE_ORDER:
                cell = agg[(agg.method == r.method) & (agg.state == s)]
                degs.append(float(cell.degradation.iloc[0]) if len(cell) else float("nan"))
            print(f"{r.method:22s}{r.coverage_breakdown:8.3f}{r.coverage_free:9.3f}"
                  f"{r.coverage_sustained:9.3f}{r.calendar_tv:6.2f}{r.degradation:8.2f}{r.bias:9.2f}"
                  + "".join(f"{d:10.2f}" for d in degs) + f"{int(r.n):4d}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(sc.DATASETS))
    parser.add_argument("--methods", nargs="+", default=[
        "random", "stride", "k_medoids", "graph_cut",
        "fd_grp10", "fd_grp05", "fd_grp00",
        "fd_hyb0802", "fd_hyb0604", "fd_strat00"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
