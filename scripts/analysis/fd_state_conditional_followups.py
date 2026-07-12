#!/usr/bin/env python3
"""Follow-up analyses for the state-conditional MAE bridge.

Consumes the archived test predictions and the FD audit, and writes three
artifacts used by the paper's FD-audit subsection:

  1. Difficulty decomposition -- splits the audit difficulty D into its
     geometry term (K * J_X) and residual (rho) per traffic state, beside the
     realised reduced-data MAE degradation.  Shows breakdown is residual
     (dynamics) dominated.
  2. Persistence baseline -- state-conditional MAE of a naive last-value
     forecast.  Directly measures intrinsic (persistence) predictability: high
     at breakdown, near free-flow at sustained congestion.  Explains why the
     audit's high congestion support difficulty does not become degradation.
  3. Paired detector test -- per detector, (breakdown degradation - free
     degradation) etc., with a Wilcoxon signed-rank test, a sign count, and a
     detector-cluster bootstrap CI, at fixed training seed 42.

Run after fd_state_conditional_mae.py has written its by-detector /
summary CSVs.  Pure post-hoc analysis; no retraining or re-inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
PHASE_C = REPO / "checkpoints" / "phase_c_method_comparison"
FITS_CSV = OUT / "fd_sensor_resolved_fits.csv"
AUDIT_CSV = OUT / "fd_sensor_resolved_by_detector.csv"
MAE_DETECTOR_CSV = OUT / "fd_state_conditional_mae_by_detector.csv"

RANGE_END = 24_192
VALID_STEPS = int(RANGE_END * 0.2)
TEST_STEPS = int(RANGE_END * 0.2)
TRAIN_STEPS = RANGE_END - VALID_STEPS - TEST_STEPS
TEST_OFFSET = TRAIN_STEPS + VALID_STEPS
T_IN = 12
NULL_VAL = 0.0
MIN_VALID = math.ceil(T_IN * 2 / 3)
MIN_AGREE = 0.75
STATES = {0: "free_to_free", 1: "breakdown", 2: "recovery", 3: "congested_to_congested"}
ORDER = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
DATASETS = ("SAN_BERNARDINO", "CONTRA_COSTA")
BACKBONE_DIR = {"STGCN": "STGCNChebGraphConv"}  # persistence uses one backbone's npz for inputs/target


def horizon_code(obs: np.ndarray) -> np.ndarray:
    T, S = obs.shape
    fp = np.vstack([np.zeros((1, S), np.int32), (obs == 0).astype(np.int32).cumsum(0)])
    cp = np.vstack([np.zeros((1, S), np.int32), (obs == 1).astype(np.int32).cumsum(0)])

    def win(starts):
        stop = starts + T_IN
        nf, nc = fp[stop] - fp[starts], cp[stop] - cp[starts]
        nv = nf + nc
        st = np.full(nf.shape, -1, np.int8)
        ok = nv >= MIN_VALID
        st[ok & (nf >= MIN_AGREE * nv)] = 0
        st[ok & (nc >= MIN_AGREE * nv)] = 1
        return st
    starts = np.arange(T - 2 * T_IN + 1)
    h, f = win(starts), win(starts + T_IN)
    code = np.full(h.shape, -1, np.int8)
    for hv, fv, c in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        code[(h == hv) & (f == fv)] = c
    return code


def test_code(dataset: str, fits: pd.DataFrame):
    acc = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor")
    sensors = acc["sensor"].to_numpy(int)
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat", dtype=np.float32, mode="r",
                    shape=tuple(desc["shape"]))
    o = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 1], np.float32)
    v = np.asarray(raw[TEST_OFFSET:RANGE_END, sensors, 2], np.float32)
    oc = acc["critical_occupancy"].to_numpy(float)[None, :]
    vc = acc["speed_at_capacity"].to_numpy(float)[None, :]
    obs = np.full(o.shape, -1, np.int8)
    obs[(o < oc) & (v > vc)] = 0
    obs[(o > oc) & (v < vc)] = 1
    return sensors, horizon_code(obs)


def detector_state_mae(pred, target, code):
    mask = target != NULL_VAL
    num = (np.abs(pred - target) * mask).sum(axis=1)
    den = mask.sum(axis=1)
    wd = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    out = {}
    for c, name in STATES.items():
        per_det = []
        for sl in range(code.shape[1]):
            col = wd[code[:, sl] == c, sl]
            col = col[np.isfinite(col)]
            if len(col):
                per_det.append(col.mean())
        out[name] = float(np.mean(per_det)) if per_det else np.nan
    return out


def decomposition(fits, method="k_medoids", ratio=0.3, k_label="K50") -> pd.DataFrame:
    b = pd.read_csv(AUDIT_CSV)
    b = b[(b.method == method) & (b.ratio == ratio) & (b.K_label == k_label)].copy()
    b["geometry"] = b["K"] * b["J_X"]
    det = (b.groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
            [["geometry", "residual", "D"]].mean())
    audit = (det.groupby(["dataset", "traffic_state_transition"], as_index=False)
                [["geometry", "residual", "D"]].mean())
    audit["residual_frac"] = audit["residual"] / audit["D"]
    # attach realised MAE degradation from the summary if present
    summ_path = OUT / "fd_state_conditional_mae_summary.csv"
    if summ_path.exists():
        s = pd.read_csv(summ_path)
        s = s[s.method == method]
        deg = (s.groupby(["dataset", "traffic_state_transition"], as_index=False)["degradation_vs_full"].mean())
        audit = audit.merge(deg, on=["dataset", "traffic_state_transition"], how="left")
    return audit


def persistence(fits) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        base = PHASE_C / BACKBONE_DIR["STGCN"] / "xtraffic" / f"{dataset}_50_12_12"
        # model_skill needs the full-data (100%-selection) run; persistence_mae
        # itself is model-free (inputs/target are identical across runs).
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
        persist = np.repeat(inp[:, -1:, :], tgt.shape[1], axis=1)
        p_mae = detector_state_mae(persist, tgt, code)
        m_mae = detector_state_mae(model, tgt, code)
        for name in ORDER:
            rows.append({"dataset": dataset, "traffic_state_transition": name,
                         "persistence_mae": p_mae[name], "model_mae": m_mae[name],
                         "model_skill_over_persistence": 1 - m_mae[name] / p_mae[name]
                         if p_mae[name] else np.nan})
    return pd.DataFrame(rows)


def paired_test(method="k_medoids", ratio=0.3, n_boot=2000) -> pd.DataFrame:
    m = pd.read_csv(MAE_DETECTOR_CSV)
    full = m[m.method == "full"][["dataset", "backbone", "sensor", "traffic_state_transition", "mae"]]
    full = full.rename(columns={"mae": "mae_full"})
    red = m[(m.method == method) & (m.ratio == ratio)][
        ["dataset", "backbone", "sensor", "traffic_state_transition", "mae"]]
    red = red.rename(columns={"mae": "mae_red"})
    j = red.merge(full, on=["dataset", "backbone", "sensor", "traffic_state_transition"])
    j["degradation"] = j["mae_red"] - j["mae_full"]
    wide = j.pivot_table(index=["dataset", "backbone", "sensor"],
                         columns="traffic_state_transition", values="degradation")
    rng = np.random.default_rng(42)
    rows = []
    for state in ("breakdown", "recovery", "congested_to_congested"):
        if state not in wide or "free_to_free" not in wide:
            continue
        diff = (wide[state] - wide["free_to_free"]).dropna().to_numpy()
        if len(diff) < 5:
            continue
        stat, p = wilcoxon(diff)
        boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
        rows.append({
            "state": state, "n_detector_cells": int(len(diff)),
            "mean_excess_degradation": float(diff.mean()),
            "median_excess_degradation": float(np.median(diff)),
            "frac_positive": float((diff > 0).mean()),
            "wilcoxon_p": float(p),
            "boot_ci_lo": float(np.quantile(boot, 0.025)),
            "boot_ci_hi": float(np.quantile(boot, 0.975)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS_CSV)

    decomp = decomposition(fits)
    decomp.to_csv(OUT / "fd_state_difficulty_decomposition.csv", index=False)
    print("=== Difficulty decomposition (k_medoids r=0.3, K50) ===")
    print(decomp.round(3).to_string(index=False), flush=True)

    pers = persistence(fits)
    pers.to_csv(OUT / "fd_state_persistence_baseline.csv", index=False)
    print("\n=== Persistence baseline vs model (STGCN) ===")
    print(pers.round(3).to_string(index=False), flush=True)

    if MAE_DETECTOR_CSV.exists():
        paired = paired_test()
        paired.to_csv(OUT / "fd_state_paired_test.csv", index=False)
        print("\n=== Paired detector test: excess degradation over free flow ===")
        print(paired.round(4).to_string(index=False), flush=True)
    else:
        print("\n[skip] paired test: run fd_state_conditional_mae.py first")

    print("\nwrote fd_state_{difficulty_decomposition,persistence_baseline,paired_test}.csv", flush=True)


if __name__ == "__main__":
    main()
