#!/usr/bin/env python3
"""The reducible margin under a seasonal baseline instead of last-value.

The margin M = e_p - e_f is currently measured against a last-value forecast held
flat over the 12-step horizon (fd_state_reducibility_multimodel.persistence_per_detector).
Flow carries a strong daily cycle, so a flat forecast is weak in exactly the states
where flow moves most, and the manuscript already attributes free flow's large margin
to persistence failing to track the demand ramp.  That makes M partly a statement
about the baseline.

This script recomputes the margin against a seasonal baseline: the training-split
mean flow of the same detector in the same (time-of-day, day-of-week) cell, which is
the standard weak baseline in traffic forecasting and absorbs the daily cycle.  If
the state ordering of M survives, M is a property of the state; if free flow's margin
collapses, M was reading the demand ramp.

Both baselines are evaluated on the same test windows, detectors and state labels as
the existing panel, so the numbers are directly comparable to Table 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fd_state_conditional_followups import (  # noqa: E402
    OUT, PHASE_C, FITS_CSV, DATASETS, BACKBONE_DIR, ORDER,
    test_code, detector_state_mae_perdet,
)
import fd_sensor_resolved as fr  # noqa: E402

REPO = fr.REPO
TRAIN_STEPS = fr.TRAIN_STEPS
RANGE_END = fr.RANGE_END
TEST_OFFSET = RANGE_END - int(RANGE_END * 0.2)
DAY = 288                      # 5-minute slots per day
JAN1_2023_WEEKDAY = 6          # Sunday, python Monday=0
NULL_VAL = 0.0


def raw_flow(dataset: str) -> np.ndarray:
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat", dtype=np.float32,
                    mode="r", shape=tuple(desc["shape"]))
    return np.asarray(raw[:, :, 0], np.float32)      # channel 0 is flow


def seasonal_table(flow_train: np.ndarray) -> np.ndarray:
    """[168*12 cells, n_sensors] mean flow by (day-of-week, time-of-day slot)."""
    t = np.arange(flow_train.shape[0])
    cell = ((t // DAY + JAN1_2023_WEEKDAY) % 7) * DAY + (t % DAY)
    n_cells = 7 * DAY
    out = np.zeros((n_cells, flow_train.shape[1]), np.float32)
    valid = flow_train != NULL_VAL
    for c in range(n_cells):
        m = cell == c
        if not m.any():
            continue
        sub, vsub = flow_train[m], valid[m]
        denom = vsub.sum(0)
        out[c] = np.where(denom > 0, (sub * vsub).sum(0) / np.maximum(denom, 1), 0.0)
    return out


def baselines(dataset: str, sensors: np.ndarray):
    """Return (persistence, seasonal, target) as [n_test_windows, 12, n_sensors]."""
    base = PHASE_C / BACKBONE_DIR["STGCN"] / "xtraffic" / f"{dataset}_50_12_12"
    npz = None
    for p in sorted(base.glob("*/*/test_results.npz")):
        cfg = (p.parent / "cfg.txt").read_text()
        if "CORESET:" not in cfg or "SELECTION_RATIO: 1.0" in cfg:
            npz = p
            break
    if npz is None:
        npz = next(base.glob("*/*/test_results.npz"))
    d = np.load(npz)
    inp = np.asarray(d["inputs"][:, :, sensors, 0])
    tgt = np.asarray(d["target"][:, :, sensors, 0])
    persist = np.repeat(inp[:, -1:, :], tgt.shape[1], axis=1)

    flow = raw_flow(dataset)
    table = seasonal_table(flow[:TRAIN_STEPS])[:, sensors]

    # test window w covers absolute steps [TEST_OFFSET+w, TEST_OFFSET+w+24);
    # the target is the last 12 of those.
    n_w, horizon = tgt.shape[0], tgt.shape[1]
    starts = TEST_OFFSET + np.arange(n_w) + fr.T_IN
    abs_t = starts[:, None] + np.arange(horizon)[None, :]
    cell = ((abs_t // DAY + JAN1_2023_WEEKDAY) % 7) * DAY + (abs_t % DAY)
    seasonal = table[cell]                                   # [n_w, 12, n_sensors]
    return persist, seasonal, tgt


def main() -> None:
    fits = pd.read_csv(FITS_CSV)
    frames = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        persist, seasonal, tgt = baselines(dataset, sensors)
        print(f"[{dataset}] {len(sensors)} detectors, {tgt.shape[0]} test windows", flush=True)
        p = detector_state_mae_perdet(persist, tgt, code, sensors).rename(
            columns={"mae": "persist_mae"})
        s = detector_state_mae_perdet(seasonal, tgt, code, sensors).rename(
            columns={"mae": "seasonal_mae"})
        f = p.merge(s, on=[c for c in p.columns if c != "persist_mae"])
        f["dataset"] = dataset
        frames.append(f)
    base = pd.concat(frames, ignore_index=True)

    mae = pd.read_csv(OUT / "fd_state_conditional_mae_by_detector.csv")
    key = ["dataset", "sensor", "traffic_state_transition"]
    full = (mae[mae.method == "full"].groupby(key + ["backbone"], as_index=False)["mae"]
            .mean().rename(columns={"mae": "full_mae"}))
    red = (mae[(mae.method == "k_medoids") & (mae.ratio == 0.3)]
           .groupby(key + ["backbone"], as_index=False)["mae"].mean()
           .rename(columns={"mae": "red_mae"}))
    df = red.merge(full, on=key + ["backbone"]).merge(base, on=key)
    df["M_persist"] = df.persist_mae - df.full_mae
    df["M_seasonal"] = df.seasonal_mae - df.full_mae
    df["delta"] = df.red_mae - df.full_mae
    df.to_csv(OUT / "fd_state_seasonal_baseline.csv", index=False)

    agg = df.groupby("traffic_state_transition").agg(
        persist_mae=("persist_mae", "mean"), seasonal_mae=("seasonal_mae", "mean"),
        full_mae=("full_mae", "mean"), M_persist=("M_persist", "mean"),
        M_seasonal=("M_seasonal", "mean"), delta=("delta", "mean"))
    agg = agg.reindex([s for s in ORDER if s in agg.index])
    agg["share_persist"] = agg.delta / agg.M_persist
    agg["share_seasonal"] = agg.delta / agg.M_seasonal
    print("\n=== K-medoids r=0.3, averaged over five architectures and both networks ===")
    print(agg.round(3).to_string())
    print(f"\nwrote {OUT / 'fd_state_seasonal_baseline.csv'}")


if __name__ == "__main__":
    main()
