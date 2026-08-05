#!/usr/bin/env python3
"""Two model-free reducibility candidates from the literature review.

PART A  ForeCA-style forecastability (Goerg, ICML 2013).
        Omega = 1 - H_spectral, the normalised spectral entropy of a series'
        power spectrum: flat spectrum (white noise) -> Omega=0, concentrated
        spectrum (predictable) -> Omega=1.  Computed per atom on the 24-step
        (history+future) trajectory, aggregated per (detector, state).  Single
        FFT per atom -- no differencing, no forecaster.  Strictly second-order.

PART B  Optimal Causation Entropy / oCSE (Sun, Taylor & Bollt, 2015).
        Learns each detector's cross-sensor parent set with NO predefined graph,
        greedily selecting the sensor that MAXIMISES causation entropy
        I(Y_i(t); X_j(t-1) | X_selected(t-1)) -- selection by maximisation, not
        by subtracting two MI estimates, so it avoids the dilution artifact.
        Causation entropy uses a Gaussian (partial-correlation) estimator here:
        fast, closed form, but linear.  The total causation entropy of the
        discovered parents is the model-free spatial reducibility of detector i.

Both are correlated against realised reduced-data degradation to test whether a
model-free signal reaches the model-based s (Spearman ~0.63).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_state_conditional_followups import (  # noqa: E402
    REPO, OUT, PHASE_C, FITS_CSV, DATASETS, BACKBONE_DIR, ORDER, NULL_VAL,
    test_code,
)

T_IN = 12
RANGE_END = 24_192
TRAIN_STEPS = RANGE_END - 2 * int(RANGE_END * 0.2)
K_PARENTS = 5          # oCSE greedy cap
MIN_GAIN = 5e-3        # nats; stop adding parents below this causation entropy


def full_data_npz(dataset: str):
    base = PHASE_C / BACKBONE_DIR["STGCN"] / "xtraffic" / f"{dataset}_50_12_12"
    for p in sorted(base.glob("*/*/test_results.npz")):
        cfg = (p.parent / "cfg.txt").read_text()
        if "CORESET:" not in cfg or "SELECTION_RATIO: 1.0" in cfg:
            return p
    return next(base.glob("*/*/test_results.npz"))


# ----------------------------- PART A: ForeCA ---------------------------------
def spectral_forecastability(traj: np.ndarray) -> np.ndarray:
    """Omega = 1 - normalised spectral entropy, along the last axis.

    traj: (..., L) real trajectories.  Detrends (removes the mean), takes the
    one-sided power spectrum excluding DC, normalises to a distribution, and
    returns 1 - H/log(nbins).
    """
    x = traj - traj.mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    p = p[..., 1:]                                   # drop DC
    tot = p.sum(-1, keepdims=True)
    p = np.divide(p, tot, out=np.full_like(p, np.nan), where=tot > 0)
    nb = p.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=-1) / np.log(nb)
    return 1.0 - h                                   # Omega in [0, 1]


def foreca_by_state() -> pd.DataFrame:
    fits = pd.read_csv(FITS_CSV)
    rows, win_rows = [], []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        d = np.load(full_data_npz(dataset))
        inp = np.asarray(d["inputs"][:, :, sensors, 0])       # (n, 12, S)
        tgt = np.asarray(d["target"][:, :, sensors, 0])
        traj = np.concatenate([inp, tgt], axis=1)             # (n, 24, S)
        traj = np.transpose(traj, (0, 2, 1))                  # (n, S, 24)
        omega = spectral_forecastability(traj)                # (n, S)
        valid = np.all(traj != NULL_VAL, axis=-1)             # fully-observed atoms
        # per-window score = mean forecastability over that window's sensors
        for i in range(omega.shape[0]):
            v = omega[i, valid[i]]
            if v.size:
                win_rows.append({"dataset": dataset, "window": i,
                                 "omega_window": float(v.mean())})
        # per (detector, state)
        for c, name in enumerate(ORDER):
            for sl in range(len(sensors)):
                sel = (code[:, sl] == c) & valid[:, sl]
                if sel.sum() < 20:
                    continue
                rows.append({"dataset": dataset, "sensor": int(sensors[sl]),
                             "traffic_state_transition": name,
                             "omega": float(omega[sel, sl].mean()),
                             "n": int(sel.sum())})
    return pd.DataFrame(rows), pd.DataFrame(win_rows)


# ----------------------------- PART B: oCSE -----------------------------------
def gaussian_cmi(ry: np.ndarray, rx: np.ndarray) -> float:
    """Gaussian conditional MI from residuals: -0.5 log(1 - partialcorr^2)."""
    a = ry - ry.mean()
    b = rx - rx.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return 0.0
    r = float((a * b).sum() / denom)
    r = min(max(r, -0.999999), 0.999999)
    return -0.5 * np.log(1.0 - r * r)


def residualise(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residual of columns of Y after regressing on [1, C]."""
    A = np.concatenate([np.ones((C.shape[0], 1)), C], axis=1)
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return Y - A @ beta


def ocse_causation_entropy() -> pd.DataFrame:
    """Per identifiable detector: total causation entropy of the greedily
    discovered cross-sensor parent set, using lag-1 predictors on the TRAIN
    split.  Model-free (Gaussian), adjacency-free, maximisation-based."""
    fits = pd.read_csv(FITS_CSV)
    rows = []
    for dataset in DATASETS:
        acc = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor")
        targets = acc["sensor"].to_numpy(int)
        desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
        raw = np.memmap(REPO / "datasets" / dataset / "data.dat", dtype=np.float32,
                        mode="r", shape=tuple(desc["shape"]))
        flow = np.asarray(raw[:TRAIN_STEPS, :, 0], np.float64)     # (T, S_all)
        valid = flow != NULL_VAL
        mean = np.divide((flow * valid).sum(0), valid.sum(0),
                         out=np.zeros(flow.shape[1]), where=valid.sum(0) > 0)
        std = np.sqrt(np.divide(((flow - mean) ** 2 * valid).sum(0), valid.sum(0),
                                out=np.ones(flow.shape[1]), where=valid.sum(0) > 0)) + 1e-6
        F = np.where(valid, (flow - mean) / std, 0.0)              # standardised
        past = F[:-1]                                              # X(t-1): (T-1, S_all)
        fut = F[1:]                                                # Y(t)
        S_all = F.shape[1]
        for it, i in enumerate(targets):
            y = fut[:, i]                                          # target future
            own = past[:, i:i + 1]                                 # own last value
            cond = own.copy()
            selected, gains = [], []
            avail = np.ones(S_all, bool); avail[i] = False
            for _ in range(K_PARENTS):
                ry = residualise(y[:, None], cond)[:, 0]
                # residualise all candidate sources on the current condition set
                rc = residualise(past, cond)                      # (T-1, S_all)
                a = ry - ry.mean()
                B = rc - rc.mean(0)
                num = (a[:, None] * B).sum(0)
                den = np.sqrt((a * a).sum() * (B * B).sum(0)) + 1e-12
                r = np.clip(num / den, -0.999999, 0.999999)
                cmi = -0.5 * np.log(1.0 - r * r)
                cmi[~avail] = -np.inf
                j = int(np.argmax(cmi))
                if not np.isfinite(cmi[j]) or cmi[j] < MIN_GAIN:
                    break
                selected.append(int(j)); gains.append(float(cmi[j]))
                cond = np.concatenate([cond, past[:, j:j + 1]], axis=1)
                avail[j] = False
            rows.append({"dataset": dataset, "sensor": int(i),
                         "n_parents": len(selected),
                         "causation_entropy": float(sum(gains)),
                         "top_gain": float(gains[0]) if gains else 0.0})
    return pd.DataFrame(rows)


# ----------------------------- reporting --------------------------------------
def main() -> None:
    red = pd.read_csv(OUT / "fd_state_reducibility.csv")

    # ---- PART A ----
    fa, fw = foreca_by_state()
    fa.to_csv(OUT / "fd_state_foreca.csv", index=False)
    g = fa.groupby("traffic_state_transition")["omega"].mean().reindex(ORDER)
    print("=== PART A  ForeCA forecastability Omega by state ===")
    print(g.round(3).to_string(), flush=True)
    m = fa.merge(red[["dataset", "sensor", "traffic_state_transition",
                      "degradation", "skill"]],
                 on=["dataset", "sensor", "traffic_state_transition"], how="inner")
    thin = m[m.traffic_state_transition != "free_to_free"].dropna(subset=["omega", "degradation"])
    r1, p1 = spearmanr(thin.omega, thin.degradation)
    r2, p2 = spearmanr(thin.skill, thin.degradation)
    print(f"\n  degradation vs Omega (non-free, n={len(thin)}): Spearman={r1:+.3f} (p={p1:.1e})")
    print(f"  degradation vs model-based s (ref)           : Spearman={r2:+.3f} (p={p2:.1e})")
    print("  within-state Spearman(degradation, Omega):")
    for st in ORDER:
        sub = m[m.traffic_state_transition == st].dropna(subset=["omega", "degradation"])
        if len(sub) < 8:
            print(f"    {st:24s}   -- ({len(sub)})"); continue
        rr, _ = spearmanr(sub.omega, sub.degradation)
        print(f"    {st:24s} {rr:+.3f} (n={len(sub)})")

    # ---- PART B ----
    ce = ocse_causation_entropy()
    ce.to_csv(OUT / "fd_detector_ocse.csv", index=False)
    print("\n=== PART B  oCSE causation entropy (per detector) ===")
    print(f"  mean parents={ce.n_parents.mean():.2f}, "
          f"mean causation entropy={ce.causation_entropy.mean():.3f} nats", flush=True)
    # per-detector degradation and s, averaged over states
    det = (red.groupby(["dataset", "sensor"], as_index=False)
              .agg(degradation=("degradation", "mean"), skill=("skill", "mean")))
    mb = ce.merge(det, on=["dataset", "sensor"], how="inner").dropna(
        subset=["causation_entropy", "degradation"])
    rc, pc = spearmanr(mb.causation_entropy, mb.degradation)
    rt, pt = spearmanr(mb.top_gain, mb.degradation)
    rs, ps = spearmanr(mb.skill, mb.degradation)
    print(f"  degradation vs causation entropy (n={len(mb)}): Spearman={rc:+.3f} (p={pc:.1e})")
    print(f"  degradation vs top-parent gain             : Spearman={rt:+.3f} (p={pt:.1e})")
    print(f"  degradation vs model-based s (ref)         : Spearman={rs:+.3f} (p={ps:.1e})")
    rcs, _ = spearmanr(mb.causation_entropy, mb.skill)
    print(f"  causation entropy vs model-based s         : Spearman={rcs:+.3f}")
    print("\nwrote fd_state_foreca.csv, fd_detector_ocse.csv", flush=True)


if __name__ == "__main__":
    main()
