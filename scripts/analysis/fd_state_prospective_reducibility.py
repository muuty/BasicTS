#!/usr/bin/env python3
"""Model-free (pre-training) reducibility references for state-conditional degradation.

Adds two references beyond last-value persistence, both computable WITHOUT training
a forecaster (profiles are built from the TRAIN split only):

  seasonal_tod      - per (detector, time-of-day) mean flow
  seasonal_tod_dow  - per (detector, time-of-day, day-of-week) mean flow

Reported per (dataset, sensor, traffic_state):
  persist_mae, seasonal_*_mae, model_mae (full-data STGCN)
  s_persist  = 1 - model/persist      (current reducibility; post-hoc)
  s_seasonal = 1 - model/seasonal     (reducibility against a stronger reference)
  s_free     = 1 - seasonal/persist   (MODEL-FREE, prospective signal)

Rationale: the degradation ceiling  deg <= R(f0) - R(f_full)  holds for any fixed
f0 in the class, so a stronger f0 tightens it. Persistence is structurally bad at
transitions, which inflates skill at breakdown for a reason unrelated to learnable
structure. A seasonal profile removes the recurring-cycle part of that margin.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_state_conditional_followups import (  # noqa: E402
    REPO, OUT, PHASE_C, FITS_CSV, DATASETS, BACKBONE_DIR,
    TEST_OFFSET, TRAIN_STEPS, RANGE_END, T_IN, NULL_VAL,
    test_code, detector_state_mae_perdet,
)

STEPS_PER_DAY = 288


def seasonal_profiles(dataset: str, sensors: np.ndarray):
    """Time-of-day and time-of-day x day-of-week mean flow from the TRAIN split."""
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat", dtype=np.float32, mode="r",
                    shape=tuple(desc["shape"]))
    flow = np.asarray(raw[:TRAIN_STEPS, sensors, 0], np.float32)  # (T_train, S)
    t = np.arange(TRAIN_STEPS)
    tod = t % STEPS_PER_DAY
    dow = (t // STEPS_PER_DAY) % 7

    valid = flow != NULL_VAL  # zeros are missing observations

    def profile(keys, n_keys):
        num = np.zeros((n_keys, flow.shape[1]), np.float64)
        den = np.zeros((n_keys, flow.shape[1]), np.float64)
        np.add.at(num, keys, np.where(valid, flow, 0.0))
        np.add.at(den, keys, valid.astype(np.float64))
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        # fall back to the detector's global mean where a cell is empty
        gmean = np.divide((flow * valid).sum(0), valid.sum(0),
                          out=np.zeros(flow.shape[1]), where=valid.sum(0) > 0)
        idx = np.isnan(out)
        out[idx] = np.broadcast_to(gmean, out.shape)[idx]
        return out

    p_tod = profile(tod, STEPS_PER_DAY)                      # (288, S)
    p_td = profile(dow * STEPS_PER_DAY + tod, 7 * STEPS_PER_DAY)  # (7*288, S)
    return p_tod, p_td


def seasonal_prediction(n_windows: int, p_tod, p_td):
    """Predict the future 12 steps of each test window from the train-split profile."""
    # window i: future absolute times TEST_OFFSET + i + T_IN + j, j = 0..T_IN-1
    i = np.arange(n_windows)[:, None]
    j = np.arange(T_IN)[None, :]
    abst = TEST_OFFSET + i + T_IN + j                 # (n, 12)
    tod = abst % STEPS_PER_DAY
    dow = (abst // STEPS_PER_DAY) % 7
    return p_tod[tod], p_td[dow * STEPS_PER_DAY + tod]  # each (n, 12, S)


def main() -> None:
    fits = pd.read_csv(FITS_CSV)
    frames = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
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
        model = np.asarray(d["prediction"][:, :, sensors, 0])
        assert tgt.shape[0] == code.shape[0], f"window misalignment {tgt.shape[0]} vs {code.shape[0]}"

        persist = np.repeat(inp[:, -1:, :], tgt.shape[1], axis=1)
        p_tod, p_td = seasonal_profiles(dataset, sensors)
        s_tod, s_td = seasonal_prediction(tgt.shape[0], p_tod, p_td)

        parts = {
            "persist_mae": persist,
            "seasonal_tod_mae": s_tod,
            "seasonal_tod_dow_mae": s_td,
            "model_mae": model,
        }
        merged = None
        for name, pred in parts.items():
            f = detector_state_mae_perdet(pred, tgt, code, sensors).rename(columns={"mae": name})
            merged = f if merged is None else merged.merge(
                f, on=["sensor", "traffic_state_transition"])
        merged["dataset"] = dataset
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out["s_persist"] = 1 - out.model_mae / out.persist_mae
    out["s_seasonal_tod"] = 1 - out.model_mae / out.seasonal_tod_mae
    out["s_seasonal_tod_dow"] = 1 - out.model_mae / out.seasonal_tod_dow_mae
    out["s_free_tod"] = 1 - out.seasonal_tod_mae / out.persist_mae
    out["s_free_tod_dow"] = 1 - out.seasonal_tod_dow_mae / out.persist_mae
    out.to_csv(OUT / "fd_state_prospective_reducibility.csv", index=False)
    print("wrote", OUT / "fd_state_prospective_reducibility.csv")

    ORDER = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
    g = out.groupby("traffic_state_transition").mean(numeric_only=True).loc[ORDER]
    print("\n=== state-level means (detector-averaged) ===")
    print(g[["persist_mae", "seasonal_tod_mae", "seasonal_tod_dow_mae", "model_mae"]].round(2))
    print("\n=== reducibility against each reference ===")
    print(g[["s_persist", "s_seasonal_tod", "s_seasonal_tod_dow",
             "s_free_tod", "s_free_tod_dow"]].round(3))


if __name__ == "__main__":
    main()
