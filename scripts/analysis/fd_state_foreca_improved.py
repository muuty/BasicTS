#!/usr/bin/env python3
"""Improve the model-free forecastability reducibility metric.

Baseline: Omega_levels = 1 - spectral entropy of the 24-step trajectory
(fd_state_forecastability_ocse.py), degradation Spearman ~+0.475.

Improvements tested here (all model-free, no kNN/binning/forecaster/differencing):
  Omega_incr : spectral forecastability of the FIRST-DIFFERENCED trajectory.
               Persistence is optimal iff increments are white, so structure in
               the increments = predictability BEYOND persistence = the paper's
               reducibility definition. A data transform, not an MI subtraction.
  fpe        : 1 - permutation entropy (Bandt-Pompe ordinal patterns, m=3).
               Model-free NONLINEAR predictability; uses ranks, not value bins.
  combined   : in-sample multiple-rank fit of degradation on the three features,
               reported as an indicative ceiling for the model-free family.
"""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_state_conditional_followups import (  # noqa: E402
    OUT, FITS_CSV, DATASETS, ORDER, NULL_VAL, test_code,
)
from fd_state_forecastability_ocse import full_data_npz, spectral_forecastability  # noqa: E402


def perm_forecastability(traj: np.ndarray, m: int = 3) -> np.ndarray:
    """1 - normalised permutation entropy along the last axis (Bandt & Pompe)."""
    L = traj.shape[-1]
    nwin = L - m + 1
    # sliding windows -> ordinal pattern id per window
    idx = np.arange(nwin)[:, None] + np.arange(m)[None, :]
    w = traj[..., idx]                                  # (..., nwin, m)
    order = np.argsort(w, axis=-1)                      # ordinal pattern
    perms = list(permutations(range(m)))
    code = np.zeros(order.shape[:-1], dtype=np.int64)   # (..., nwin)
    for pid, perm in enumerate(perms):
        code[np.all(order == np.array(perm), axis=-1)] = pid
    npat = len(perms)
    # histogram over the nwin axis -> entropy
    oh = np.eye(npat, dtype=np.float64)[code]           # (..., nwin, npat)
    p = oh.sum(-2)
    p = p / p.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), -1) / np.log(npat)
    return 1.0 - h


def build() -> pd.DataFrame:
    fits = pd.read_csv(FITS_CSV)
    rows = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        d = np.load(full_data_npz(dataset))
        inp = np.asarray(d["inputs"][:, :, sensors, 0])
        tgt = np.asarray(d["target"][:, :, sensors, 0])
        traj = np.transpose(np.concatenate([inp, tgt], 1), (0, 2, 1))  # (n, S, 24)
        valid = np.all(traj != NULL_VAL, axis=-1)
        om_lv = spectral_forecastability(traj)
        om_in = spectral_forecastability(np.diff(traj, axis=-1))
        fpe = perm_forecastability(traj, m=3)
        for c, name in enumerate(ORDER):
            for sl in range(len(sensors)):
                sel = (code[:, sl] == c) & valid[:, sl]
                if sel.sum() < 20:
                    continue
                rows.append({"dataset": dataset, "sensor": int(sensors[sl]),
                             "traffic_state_transition": name, "n": int(sel.sum()),
                             "omega_levels": float(om_lv[sel, sl].mean()),
                             "omega_incr": float(om_in[sel, sl].mean()),
                             "fpe": float(fpe[sel, sl].mean())})
    return pd.DataFrame(rows)


def rank_combo(df, feats, target):
    """In-sample multiple correlation on ranks: predicted vs actual Spearman."""
    R = np.column_stack([pd.Series(df[f]).rank().to_numpy() for f in feats])
    R = (R - R.mean(0)) / (R.std(0) + 1e-9)
    y = pd.Series(df[target]).rank().to_numpy()
    y = (y - y.mean()) / (y.std() + 1e-9)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), R]), y, rcond=None)
    yhat = np.column_stack([np.ones(len(y)), R]) @ beta
    return spearmanr(yhat, df[target])[0]


def main() -> None:
    df = build()
    df.to_csv(OUT / "fd_state_foreca_improved.csv", index=False)
    red = pd.read_csv(OUT / "fd_state_reducibility.csv")
    m = df.merge(red[["dataset", "sensor", "traffic_state_transition",
                      "degradation", "skill"]],
                 on=["dataset", "sensor", "traffic_state_transition"], how="inner")

    print("=== state-mean forecastability ===")
    print(m.groupby("traffic_state_transition")[["omega_levels", "omega_incr", "fpe"]]
          .mean().reindex(ORDER).round(3).to_string(), flush=True)

    thin = m[m.traffic_state_transition != "free_to_free"].dropna(
        subset=["omega_levels", "omega_incr", "fpe", "degradation"])
    print(f"\n=== degradation Spearman (non-free, n={len(thin)}) ===")
    for f in ["omega_levels", "omega_incr", "fpe", "skill"]:
        r, p = spearmanr(thin[f], thin.degradation)
        print(f"  {f:16s}: {r:+.3f} (p={p:.1e})")
    combo = rank_combo(thin, ["omega_levels", "omega_incr", "fpe"], "degradation")
    print(f"  {'combined(3)':16s}: {combo:+.3f}  (in-sample ceiling)")

    print("\n=== within-state Spearman(degradation, .) ===")
    print(f"  {'state':22s} {'lv':>7s} {'incr':>7s} {'fpe':>7s} {'s':>7s}   n")
    for st in ORDER:
        sub = m[m.traffic_state_transition == st].dropna(
            subset=["omega_levels", "omega_incr", "fpe", "degradation"])
        if len(sub) < 8:
            print(f"  {st:22s}   -- ({len(sub)})"); continue
        vals = [spearmanr(sub[f], sub.degradation)[0]
                for f in ["omega_levels", "omega_incr", "fpe", "skill"]]
        print(f"  {st:22s} " + " ".join(f"{v:+7.3f}" for v in vals) + f"  {len(sub)}")
    print("\nwrote fd_state_foreca_improved.csv", flush=True)


if __name__ == "__main__":
    main()
