#!/usr/bin/env python3
"""Agreement between the fitted-FD traffic states and the conventional definition.

The article labels a detector congested when occupancy is above its fitted
critical occupancy and speed is below the median speed of the critical bin.
Both parts come from a per-detector fit whose acceptance thresholds are
conventions of ours.  The traffic-engineering literature labels breakdown from
speed alone, with a fixed threshold and a minimum duration:

  Brilon, Geistefeldt and Regler (2005) take 70 km/h, about 45 mph, as the
  speed below which a freeway section counts as broken down.
  Dehman and Drakopoulos (2012) require the drop to persist for 15 minutes,
  three intervals at this resolution.
  Chen, Skabardonis and Varaiya (2004) use 40 mph on PeMS five-minute speeds,
  with a downstream condition this script does not apply.

This script builds the conventional labels, aggregates them into the same
12-step history and future segments the article uses, and reports how far the
two labellings agree.  With ``--stage mae`` it then recomputes the
state-resolved test MAE under both labellings from the archived predictions, so
the article's headline can be read off the conventional labels without
retraining anything.

Usage:
    python3 fd_label_agreement.py --stage labels
    python3 fd_label_agreement.py --stage mae --methods random
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fd_state_conditional_mae import (
    BACKBONE_DIRS,
    DATASETS,
    FITS_CSV,
    FULL_KEY,
    OUT,
    RANGE_END,
    REPO,
    STATE_NAMES,
    T_IN,
    TEST_OFFSET,
    TRAIN_STEPS,
    detector_state_rows,
    horizon_transition,
    leaf_map,
    window_detector_mae,
)

# (name, mode, level, persistence in five-minute intervals)
#   absolute levels are mph thresholds taken from the cited work
#   relative levels are fractions of the detector's own free-flow speed
VARIANTS = (
    ("brilon45_15min", "absolute", 45.0, 3),
    ("brilon45_5min", "absolute", 45.0, 1),
    ("chen40_15min", "absolute", 40.0, 3),
    ("relative75_15min", "relative", 0.75, 3),
    ("relative75_5min", "relative", 0.75, 1),
    ("relative85_15min", "relative", 0.85, 3),
)
FFS_PERCENTILE = 85.0  # free-flow speed of a detector, the usual PeMS convention


def open_runs(mask: np.ndarray, persist: int) -> np.ndarray:
    """Keep only entries lying in a run of at least ``persist`` consecutive True.

    A morphological opening along time, written with prefix sums so it stays
    vectorised over detectors.
    """
    if persist <= 1:
        return mask
    t = mask.shape[0]
    if t < persist:
        return np.zeros_like(mask)
    csum = np.vstack([np.zeros((1, mask.shape[1]), np.int32), mask.cumsum(0, dtype=np.int32)])
    starts = np.arange(t - persist + 1)
    full = (csum[starts + persist] - csum[starts]) == persist  # run starts here
    out = np.zeros_like(mask)
    for offset in range(persist):
        out[starts + offset] |= full
    return out


def free_flow_speed(dataset: str, sensors: np.ndarray) -> np.ndarray:
    """Per-detector free-flow speed from the training split alone."""
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat",
                    dtype=np.float32, mode="r", shape=tuple(desc["shape"]))
    speed = np.asarray(raw[:TRAIN_STEPS, sensors, 2], np.float32)
    speed = np.where(speed > 0, speed, np.nan)
    return np.nanpercentile(speed, FFS_PERCENTILE, axis=0)


def conventional_observations(dataset: str, sensors: np.ndarray, mode: str,
                              level: float, persist: int) -> np.ndarray:
    """Test-split observation states under a speed threshold with persistence."""
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat",
                    dtype=np.float32, mode="r", shape=tuple(desc["shape"]))
    speed = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 2], np.float32)
    if mode == "absolute":
        threshold = np.full(len(sensors), level, dtype=np.float32)[None, :]
    else:
        threshold = (level * free_flow_speed(dataset, sensors)).astype(np.float32)[None, :]
    valid = np.isfinite(speed) & (speed > 0)
    congested = open_runs(valid & (speed < threshold), persist)
    obs = np.full(speed.shape, -1, np.int8)
    obs[valid & ~congested] = 0
    obs[congested] = 1
    return obs


def fd_observations(dataset: str, accepted: pd.DataFrame) -> np.ndarray:
    """Test-split observation states under the article's fitted-FD rule."""
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat",
                    dtype=np.float32, mode="r", shape=tuple(desc["shape"]))
    sensors = accepted["sensor"].to_numpy(int)
    occ = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 1], np.float32)
    speed = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 2], np.float32)
    oc = accepted["critical_occupancy"].to_numpy(float)[None, :]
    vc = accepted["speed_at_capacity"].to_numpy(float)[None, :]
    obs = np.full(occ.shape, -1, np.int8)
    obs[(occ < oc) & (speed > vc)] = 0
    obs[(occ > oc) & (speed < vc)] = 1
    return obs


def kappa(a: np.ndarray, b: np.ndarray, n_classes: int) -> tuple[float, float, np.ndarray]:
    """Cohen's kappa, raw agreement and the confusion matrix over paired labels."""
    table = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(table, (a, b), 1)
    total = table.sum()
    if total == 0:
        return float("nan"), float("nan"), table
    observed = np.trace(table) / total
    expected = float((table.sum(1) @ table.sum(0)) / total ** 2)
    return (observed - expected) / (1 - expected), observed, table


def stage_labels(args: argparse.Namespace) -> None:
    fits = pd.read_csv(FITS_CSV)
    min_valid = int(math.ceil(T_IN * args.min_valid_fraction))
    rows = []
    for dataset in args.datasets:
        accepted = fits[(fits.dataset == dataset) & fits.identifiable] \
            .sort_values("sensor").reset_index(drop=True)
        sensors = accepted["sensor"].to_numpy(int)
        vc = accepted["speed_at_capacity"].to_numpy(float)
        ffs = free_flow_speed(dataset, sensors)
        print(f"\n===== {dataset}, {len(sensors)} detectors =====")
        print(f"  fitted critical-bin speed: median {np.median(vc):.1f} mph, "
              f"quartiles {np.percentile(vc, 25):.1f} and {np.percentile(vc, 75):.1f}, "
              f"below 45 mph at {100 * (vc < 45).mean():.0f}% of detectors")
        print(f"  free-flow speed (p{FFS_PERCENTILE:.0f} of training speed): "
              f"median {np.median(ffs):.1f} mph, implied ratio "
              f"median {np.median(vc / ffs):.2f}")

        fd_obs = fd_observations(dataset, accepted)
        fd_code = horizon_transition(fd_obs, min_valid, args.min_agreement)

        for name, mode, level, persist in VARIANTS:
            cv_obs = conventional_observations(dataset, sensors, mode, level, persist)
            cv_code = horizon_transition(cv_obs, min_valid, args.min_agreement)

            both = (fd_obs >= 0) & (cv_obs >= 0)
            k_obs, agree_obs, _ = kappa(fd_obs[both].astype(int), cv_obs[both].astype(int), 2)

            paired = (fd_code >= 0) & (cv_code >= 0)
            k_seg, agree_seg, table = kappa(
                fd_code[paired].astype(int), cv_code[paired].astype(int), 4)

            fd_bd = int((fd_code == 1).sum())
            cv_bd = int((cv_code == 1).sum())
            both_bd = int(((fd_code == 1) & (cv_code == 1)).sum())
            fd_rc = int((fd_code == 2).sum())
            cv_rc = int((cv_code == 2).sum())
            both_rc = int(((fd_code == 2) & (cv_code == 2)).sum())

            print(f"\n  -- {name} --")
            print(f"     observation level  agreement {agree_obs:.3f}  kappa {k_obs:.3f}"
                  f"   ({both.sum() / both.size:.3f} of pairs resolved by both)")
            print(f"     segment level      agreement {agree_seg:.3f}  kappa {k_seg:.3f}")
            print(f"     breakdown  FD {fd_bd:7d}   conventional {cv_bd:7d}   "
                  f"both {both_bd:7d}   Jaccard {both_bd / max(fd_bd + cv_bd - both_bd, 1):.3f}")
            print(f"     recovery   FD {fd_rc:7d}   conventional {cv_rc:7d}   "
                  f"both {both_rc:7d}   Jaccard {both_rc / max(fd_rc + cv_rc - both_rc, 1):.3f}")
            print("     confusion (rows FD, columns conventional), "
                  + ", ".join(STATE_NAMES[i] for i in range(4)))
            for i in range(4):
                print("       " + "".join(f"{table[i, j]:9d}" for j in range(4)))

            rows.append({
                "dataset": dataset, "variant": name, "mode": mode, "level": level,
                "persist_intervals": persist,
                "observation_agreement": agree_obs, "observation_kappa": k_obs,
                "segment_agreement": agree_seg, "segment_kappa": k_seg,
                "fd_breakdown": fd_bd, "conventional_breakdown": cv_bd,
                "shared_breakdown": both_bd,
                "fd_recovery": fd_rc, "conventional_recovery": cv_rc,
                "shared_recovery": both_rc,
                "median_critical_speed": float(np.median(vc)),
                "median_free_flow_speed": float(np.median(ffs)),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fd_label_agreement.csv", index=False)
    print(f"\nwrote {OUT / 'fd_label_agreement.csv'}")


def stage_mae(args: argparse.Namespace) -> None:
    """State-resolved MAE under both labellings, from archived predictions."""
    fits = pd.read_csv(FITS_CSV)
    min_valid = int(math.ceil(T_IN * args.min_valid_fraction))
    name, mode, level, persist = next(v for v in VARIANTS if v[0] == args.variant)
    cell_keys = [FULL_KEY] + [(m, args.ratio) for m in args.methods]

    rows: list[dict] = []
    for dataset in args.datasets:
        accepted = fits[(fits.dataset == dataset) & fits.identifiable] \
            .sort_values("sensor").reset_index(drop=True)
        sensors = accepted["sensor"].to_numpy(int)
        codes = {
            "fitted_fd": horizon_transition(
                fd_observations(dataset, accepted), min_valid, args.min_agreement),
            name: horizon_transition(
                conventional_observations(dataset, sensors, mode, level, persist),
                min_valid, args.min_agreement),
        }
        for labelling, code in codes.items():
            atoms = {s: int((code == c).sum()) for c, s in STATE_NAMES.items()}
            print(f"[{dataset}] {labelling}: {atoms}", flush=True)

        for backbone, backbone_dir in BACKBONE_DIRS.items():
            if backbone not in args.backbones:
                continue
            leaves = leaf_map(backbone_dir, dataset)
            for method, ratio in cell_keys:
                paths = leaves.get((method, ratio))
                if not paths:
                    print(f"  {backbone:11s} [skip] {method} r={ratio}", flush=True)
                    continue
                label = "full" if (method, ratio) == FULL_KEY else method
                wd_mae = window_detector_mae(paths, sensors, args.max_seeds)
                for labelling, code in codes.items():
                    for row in detector_state_rows(
                            wd_mae, code, sensors, dataset, backbone, label, ratio):
                        row["labelling"] = labelling
                        rows.append(row)
                print(f"  {backbone:11s} {label:10s} r={ratio}: "
                      f"{min(len(paths), args.max_seeds)}/{len(paths)} seeds", flush=True)

    by_detector = pd.DataFrame(rows)
    by_detector.to_csv(OUT / "fd_label_agreement_mae_by_detector.csv", index=False)

    summary = (by_detector.groupby(
        ["labelling", "dataset", "backbone", "method", "traffic_state_transition"],
        as_index=False).agg(mae=("mae", "mean"), n_detectors=("sensor", "nunique"),
                            n_windows=("n_windows", "sum")))
    full = summary[summary.method == "full"][
        ["labelling", "dataset", "backbone", "traffic_state_transition", "mae"]
    ].rename(columns={"mae": "mae_full"})
    summary = summary.merge(
        full, on=["labelling", "dataset", "backbone", "traffic_state_transition"], how="left")
    summary["degradation_vs_full"] = summary["mae"] - summary["mae_full"]
    summary.to_csv(OUT / "fd_label_agreement_mae_summary.csv", index=False)

    for method in args.methods:
        print(f"\n===== {method} at r={args.ratio}, MAE increase over full data =====")
        print(f"{'labelling':16s}{'state':26s}{'mean':>8s}{'vs free':>9s}"
              f"{'worse in':>10s}")
        for labelling in ("fitted_fd", name):
            sub = summary[(summary.labelling == labelling) & (summary.method == method)]
            pivot = sub.pivot_table(index=["dataset", "backbone"],
                                    columns="traffic_state_transition",
                                    values="degradation_vs_full")
            if pivot.empty:
                continue
            free = pivot["free_to_free"]
            for state in ("free_to_free", "breakdown", "recovery",
                          "congested_to_congested"):
                if state not in pivot:
                    continue
                diff = pivot[state] - free
                print(f"{labelling:16s}{state:26s}{pivot[state].mean():8.2f}"
                      f"{diff.median():9.2f}"
                      f"{f'{(pivot[state] > 0).sum()}/{pivot[state].notna().sum()}':>10s}")
    print(f"\nwrote {OUT / 'fd_label_agreement_mae_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("labels", "mae"), default="labels")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONE_DIRS),
                        default=list(BACKBONE_DIRS))
    parser.add_argument("--methods", nargs="+", default=["random"])
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--variant", default="brilon45_15min")
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "labels":
        stage_labels(arguments)
    else:
        stage_mae(arguments)
