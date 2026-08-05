#!/usr/bin/env python3
"""Information-theoretic reducibility, by traffic state (no kNN, no binning).

Separates the two senses of "random" that raw future divergence (rho) fuses:

  aleatoric floor   H(Y | X_full)          irreducible even with infinite data;
                                           more samples cannot help -> should NOT
                                           predict degradation.
  reducible info    I(Y; X_full | X_last)  future IS determined by history but the
                    = H(Y|X_last)          mapping is complex/rare; samples are
                      - H(Y|X_full)        valuable -> should predict degradation.

Two surrogates, both closed-form (Gaussian / Laplace), no density estimation.

PART A  Laplace surrogate on existing state-conditional MAE (persistence,
        seasonal profile, full-data model).  Differential entropy of a Laplace
        residual with scale b is 1 + log(2b), and MAE estimates b, so
            I = H_a - H_b = log(MAE_a / MAE_b).
        This splits total reducible information into a prospective seasonal
        (recurring-cycle) part and a dynamic (history) part:
            I_total(model)  = log(persist / model)          [= -log(1 - s)]
            I_seasonal      = log(persist / seasonal_td)     [model-free]
            I_dynamic       = log(seasonal_td / model)       [model beyond season]

PART B  Linear-Gaussian conditional mutual information from the raw windows.
        Under a linear-Gaussian surrogate, I(Y_h; X_full | X_last) is a log-ratio
        of residual variances from two least-squares fits per (detector, state,
        horizon): Y_h on [1, x_last] versus Y_h on [1, x_1..x_12].  Residual
        variances are df-corrected (RSS/(n-p)) so extra regressors that do not
        genuinely reduce uncertainty do not inflate I.  This is prospective:
        it uses only history->future second-order structure, no trained forecaster.

Correlates both against realised reduced-data degradation to test the split.
"""
from __future__ import annotations

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
MIN_N = 80          # minimum clean windows in a (detector, state) cell
NMAX = 1500         # subsample cap per cell (Gram matrix is n x n)
SIGMA_MULT = 1.0    # multiplier on the median-heuristic bandwidth
RENYI_ALPHA = 2.0   # order-2 Renyi: closed form via Frobenius norm, no eigdecomp
SEED = 42
HALF_LOG_2PIE = 0.5 * np.log(2 * np.pi * np.e)


def full_data_npz(dataset: str):
    base = PHASE_C / BACKBONE_DIR["STGCN"] / "xtraffic" / f"{dataset}_50_12_12"
    for p in sorted(base.glob("*/*/test_results.npz")):
        cfg = (p.parent / "cfg.txt").read_text()
        if "CORESET:" not in cfg or "SELECTION_RATIO: 1.0" in cfg:
            return p
    return next(base.glob("*/*/test_results.npz"))


# ------------------- matrix-based Renyi entropy (model-free) -------------------
# Sanchez Giraldo, Rao & Principe (2015), "Measures of Entropy from Data Using
# Infinitely Divisible Kernels."  Entropy is read off the spectrum of a normalised
# kernel Gram matrix -- no density estimation, no binning, no kNN, no forecaster.
# Purely RKHS pairwise geometry.  Order alpha=2 is closed form in the Frobenius
# norm, so no eigendecomposition is needed.

def gram(Z: np.ndarray) -> np.ndarray:
    """RBF Gram matrix with a median-heuristic bandwidth; unit diagonal."""
    sq = (Z ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices_from(d2, k=1)
    med = np.median(d2[iu])
    sig2 = SIGMA_MULT * (med if med > 0 else 1.0)
    return np.exp(-d2 / (2.0 * sig2))


def s2(K: np.ndarray) -> float:
    """Order-2 matrix Renyi entropy of A = K / tr(K):  -log(||A||_F^2)."""
    n = K.shape[0]
    return -np.log((K ** 2).sum() / (n * n))          # tr(K)=n since diag=1


def s2_joint(K1: np.ndarray, K2: np.ndarray) -> float:
    """Order-2 entropy of the normalised Hadamard product (joint variable)."""
    n = K1.shape[0]
    KK = K1 * K2
    return -np.log((KK ** 2).sum() / (n * n))         # tr(K1 o K2)=n


def renyi_terms(Xfull, Xlast, Y):
    """Matrix-Renyi dependence terms for one (detector, state) cell."""
    Kf, Kl, Ky = gram(Xfull), gram(Xlast), gram(Y)
    S_f, S_l, S_y = s2(Kf), s2(Kl), s2(Ky)
    I_full = S_y + S_f - s2_joint(Ky, Kf)             # I(Y; X_full)
    I_last = S_y + S_l - s2_joint(Ky, Kl)             # I(Y; X_last)
    H_cond_full = s2_joint(Ky, Kf) - S_f              # H(Y | X_full)
    return {
        "I_full": float(I_full),
        "I_last": float(I_last),
        "I_learnable": float(I_full - I_last),        # I(Y; X_full | X_last)
        "H_floor": float(H_cond_full),                # aleatoric floor
        "S_y": float(S_y),
    }


# ----------------------------- PART A -----------------------------------------
def laplace_decomposition() -> pd.DataFrame:
    csv = OUT / "fd_state_prospective_reducibility.csv"
    df = pd.read_csv(csv)
    eps = 1e-9
    df = df[(df.persist_mae > eps) & (df.model_mae > eps) &
            (df.seasonal_tod_dow_mae > eps)].copy()
    df["I_total"] = np.log(df.persist_mae / df.model_mae)          # -log(1-s)
    df["I_seasonal"] = np.log(df.persist_mae / df.seasonal_tod_dow_mae)
    df["I_dynamic"] = np.log(df.seasonal_tod_dow_mae / df.model_mae)
    df["H_floor_laplace"] = 1.0 + np.log(2.0 * df.model_mae)       # aleatoric floor
    return df


# ----------------------------- PART B -----------------------------------------
def matrix_renyi_mi() -> pd.DataFrame:
    """Per (detector, state): model-free matrix-based Renyi mutual information
    between the future and the history, decomposed into the last-value part and
    the learnable part I(Y; X_full | X_last) = I(Y; X_full) - I(Y; X_last),
    plus the aleatoric floor H(Y | X_full).  RBF kernels; no forecaster.
    """
    rng = np.random.default_rng(SEED)
    fits = pd.read_csv(FITS_CSV)
    rows = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        d = np.load(full_data_npz(dataset))
        inp = np.asarray(d["inputs"][:, :, sensors, 0])       # (n, 12, S)
        tgt = np.asarray(d["target"][:, :, sensors, 0])       # (n, 12, S)
        assert tgt.shape[0] == code.shape[0], "window misalignment"
        for c, name in enumerate(["free_to_free", "breakdown", "recovery",
                                  "congested_to_congested"]):
            for sl in range(len(sensors)):
                w = np.where(code[:, sl] == c)[0]
                if len(w) < MIN_N:
                    continue
                X = inp[w, :, sl]                              # (nw, 12)
                Y = tgt[w, :, sl]
                # keep windows with fully observed history and future
                ok = np.all(X != NULL_VAL, axis=1) & np.all(Y != NULL_VAL, axis=1)
                X, Y = X[ok], Y[ok]
                if len(X) < MIN_N:
                    continue
                if len(X) > NMAX:
                    idx = rng.choice(len(X), NMAX, replace=False)
                    X, Y = X[idx], Y[idx]
                # per-cell standardisation so the median bandwidth is well scaled
                Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
                Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
                t = renyi_terms(Xz, Xz[:, -1:], Yz)
                t.update({"dataset": dataset, "sensor": int(sensors[sl]),
                          "traffic_state_transition": name, "n": int(len(X))})
                rows.append(t)
    return pd.DataFrame(rows)


# ------------------- PART C: data-driven spatial conditioning ------------------
# No predefined adjacency: the network's own SVD modes (train split) define the
# co-variation structure -- the model-free analogue of a learned/adaptive graph.
# Condition each detector's future on the global latent state Z and ask how much
# it explains BEYOND the detector's own last value: I(Y_i; Z | X_i,last).
K_MODES = 10


def global_latent(dataset: str, sensors_all: int):
    import json
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(REPO / "datasets" / dataset / "data.dat", dtype=np.float32, mode="r",
                    shape=tuple(desc["shape"]))
    RANGE_END = 24_192
    VALID = int(RANGE_END * 0.2)
    TEST = int(RANGE_END * 0.2)
    TRAIN = RANGE_END - VALID - TEST
    TEST_OFF = TRAIN + VALID
    flow = np.asarray(raw[:RANGE_END, :, 0], np.float64)     # (RANGE_END, S_all)
    valid = flow != NULL_VAL
    col_mean = np.divide((flow * valid).sum(0), valid.sum(0),
                         out=np.zeros(flow.shape[1]), where=valid.sum(0) > 0)
    fc = np.where(valid, flow - col_mean, 0.0)               # centred, missing->0
    # SVD modes from the train split only
    _, _, Vt = np.linalg.svd(fc[:TRAIN], full_matrices=False)
    V = Vt[:K_MODES].T                                       # (S_all, k)
    Z_all = fc @ V                                           # (RANGE_END, k)
    return Z_all, TEST_OFF


def matrix_renyi_spatial() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    fits = pd.read_csv(FITS_CSV)
    rows = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        d = np.load(full_data_npz(dataset))
        inp = np.asarray(d["inputs"][:, :, sensors, 0])
        tgt = np.asarray(d["target"][:, :, sensors, 0])
        Z_all, TEST_OFF = global_latent(dataset, len(sensors))
        # window i's last-history step is at absolute time TEST_OFF + i + T_IN - 1
        widx = np.arange(tgt.shape[0]) + TEST_OFF + T_IN - 1
        Zwin = Z_all[widx]                                   # (n_windows, k)
        for c, name in enumerate(["free_to_free", "breakdown", "recovery",
                                  "congested_to_congested"]):
            for sl in range(len(sensors)):
                w = np.where(code[:, sl] == c)[0]
                if len(w) < MIN_N:
                    continue
                X = inp[w, :, sl]
                Y = tgt[w, :, sl]
                ok = np.all(X != NULL_VAL, axis=1) & np.all(Y != NULL_VAL, axis=1)
                w2 = w[ok]; X, Y = X[ok], Y[ok]
                if len(X) < MIN_N:
                    continue
                if len(X) > NMAX:
                    idx = rng.choice(len(X), NMAX, replace=False)
                    w2 = w2[idx]; X, Y = X[idx], Y[idx]
                last = X[:, -1:]
                Z = Zwin[w2]                                 # (m, k)
                Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
                lz = (last - last.mean(0)) / (last.std(0) + 1e-6)
                Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
                Ky, Kl, Kzl = gram(Yz), gram(lz), gram(np.concatenate([Zz, lz], 1))
                S_y, S_l, S_zl = s2(Ky), s2(Kl), s2(Kzl)
                I_last = S_y + S_l - s2_joint(Ky, Kl)             # I(Y; X_last)
                I_zlast = S_y + S_zl - s2_joint(Ky, Kzl)         # I(Y; Z, X_last)
                rows.append({
                    "dataset": dataset, "sensor": int(sensors[sl]),
                    "traffic_state_transition": name, "n": int(len(X)),
                    "I_last": float(I_last),
                    "I_spatial": float(I_zlast - I_last),          # I(Y; Z | X_last)
                    "I_joint": float(I_zlast),
                })
    return pd.DataFrame(rows)


def state_table(df: pd.DataFrame, cols, label: str):
    g = df.groupby("traffic_state_transition")[cols].mean().reindex(ORDER)
    print(f"\n=== {label} (detector-averaged, per state) ===")
    print(g.round(3).to_string(), flush=True)
    return g


def main() -> None:
    # ---- PART A ----
    a = laplace_decomposition()
    ga = state_table(a, ["I_total", "I_seasonal", "I_dynamic", "H_floor_laplace"],
                     "PART A  Laplace-surrogate reducible information [nats]")
    frac = (ga["I_dynamic"] / ga["I_total"]).round(3)
    print("\n  dynamic fraction  I_dynamic / I_total:")
    print(frac.to_string(), flush=True)

    # ---- PART B  (model-free matrix-based Renyi) ----
    b = matrix_renyi_mi()
    b.to_csv(OUT / "fd_state_information_reducibility.csv", index=False)
    cov = b.groupby("traffic_state_transition").size().reindex(ORDER)
    print("\n  detector cells surviving MIN_N:", dict(cov.fillna(0).astype(int)))
    state_table(b, ["I_full", "I_last", "I_learnable", "H_floor", "S_y"],
                "PART B  Matrix-Renyi MI (model-free, alpha=2) [nats]")

    # ---- decisive test: which quantity predicts per-detector degradation? ----
    red = pd.read_csv(OUT / "fd_state_reducibility.csv")
    m = b.merge(red[["dataset", "sensor", "traffic_state_transition",
                     "degradation", "skill"]],
                on=["dataset", "sensor", "traffic_state_transition"], how="inner")
    thin = m[m.traffic_state_transition != "free_to_free"].dropna(
        subset=["I_learnable", "H_floor", "degradation"])
    print(f"\n=== Decisive test: predicting per-detector degradation "
          f"(non-free states, n={len(thin)}) ===")
    for col, desc in [("I_learnable", "learnable info I(Y;X_full|X_last)"),
                      ("I_full", "total dependence I(Y;X_full)"),
                      ("H_floor", "aleatoric floor H(Y|X_full)"),
                      ("skill", "current model-based s")]:
        r, p = spearmanr(thin[col], thin["degradation"])
        print(f"  degradation vs {desc:38s}: Spearman={r:+.3f} (p={p:.1e})")

    # within-state, to strip the between-state confound
    print("\n  within-state Spearman(degradation, .):")
    print(f"    {'state':24s} {'learn':>8s} {'floor':>8s} {'s':>8s}   n")
    for st in ORDER:
        sub = m[m.traffic_state_transition == st].dropna(subset=["I_learnable", "degradation"])
        if len(sub) < 8:
            print(f"    {st:24s} {'--':>8s} {'--':>8s} {'--':>8s}  {len(sub)}")
            continue
        ri, _ = spearmanr(sub.I_learnable, sub.degradation)
        rf, _ = spearmanr(sub.H_floor, sub.degradation)
        rs, _ = spearmanr(sub.skill, sub.degradation)
        print(f"    {st:24s} {ri:+8.3f} {rf:+8.3f} {rs:+8.3f}  {len(sub)}")

    # agreement of the model-free learnable info with the current model-based s
    both = m.dropna(subset=["I_learnable", "skill"])
    r, p = spearmanr(both.I_learnable, both.skill)
    print(f"\n  I_learnable vs current model-based s: Spearman={r:+.3f} "
          f"(p={p:.1e}, n={len(both)})")

    # ---- PART C  (data-driven spatial conditioning, no adjacency) ----
    sp = matrix_renyi_spatial()
    sp.to_csv(OUT / "fd_state_spatial_reducibility.csv", index=False)
    state_table(sp, ["I_last", "I_spatial", "I_joint"],
                "PART C  Spatial MI I(Y; Z_network | X_last), SVD latent [nats]")
    ms = sp.merge(red[["dataset", "sensor", "traffic_state_transition",
                       "degradation", "skill"]],
                  on=["dataset", "sensor", "traffic_state_transition"], how="inner")
    thin = ms[ms.traffic_state_transition != "free_to_free"].dropna(
        subset=["I_spatial", "degradation"])
    print(f"\n=== Decisive test: does spatial info predict degradation? "
          f"(non-free, n={len(thin)}) ===")
    for col, desc in [("I_spatial", "spatial info I(Y;Z|X_last)"),
                      ("I_joint", "joint info I(Y;Z,X_last)"),
                      ("I_last", "own last-value I(Y;X_last)"),
                      ("skill", "current model-based s")]:
        rr, pp = spearmanr(thin[col], thin["degradation"])
        print(f"  degradation vs {desc:34s}: Spearman={rr:+.3f} (p={pp:.1e})")
    print("\n  within-state Spearman(degradation, I_spatial):")
    for st in ORDER:
        sub = ms[ms.traffic_state_transition == st].dropna(subset=["I_spatial", "degradation"])
        if len(sub) < 8:
            print(f"    {st:24s}   --   ({len(sub)})"); continue
        rr, _ = spearmanr(sub.I_spatial, sub.degradation)
        print(f"    {st:24s} {rr:+.3f}  (n={len(sub)})")
    r, p = spearmanr(ms.dropna(subset=["I_spatial", "skill"]).I_spatial,
                     ms.dropna(subset=["I_spatial", "skill"]).skill)
    print(f"\n  I_spatial vs current model-based s: Spearman={r:+.3f} (p={p:.1e})")

    print("\nwrote", OUT / "fd_state_information_reducibility.csv",
          "and fd_state_spatial_reducibility.csv", flush=True)


if __name__ == "__main__":
    main()
