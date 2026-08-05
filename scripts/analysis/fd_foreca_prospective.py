#!/usr/bin/env python3
"""Analysis B: leakage-free, length-matched prospective forecastability + clustered CI.

The headline Omega (+0.475) was computed on the 24-step history+FUTURE trajectory, and
degradation is measured on that same future -> look-ahead leakage. This isolates the
honest number by computing Omega on a 24-step PURE-HISTORY window ending at the forecast
origin (pulled from raw data.dat), so it is (a) leakage-free and (b) length-matched to the
leaky version, separating the leakage effect from the 12-vs-24 spectral-length artifact.

Variants per atom:
  omega_hf24 : 24-step history+future (npz)              -> reproduces the leaky +0.475
  omega_h12  : 12-step history only   (npz inputs)       -> reviewer's +0.295
  omega_h24  : 24-step pure history ending at origin (raw)-> THE honest, length-matched number

Inference: Spearman with degradation, with a DETECTOR-CLUSTERED bootstrap 95% CI
(resample detectors, not cells), plus per-dataset and within-state breakdowns.
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
    REPO, OUT, FITS_CSV, DATASETS, ORDER, NULL_VAL,
    RANGE_END, TEST_OFFSET, T_IN, test_code,
)
from fd_state_forecastability_ocse import full_data_npz, spectral_forecastability  # noqa: E402

SEED = 42
N_BOOT = 2000


def build() -> pd.DataFrame:
    fits = pd.read_csv(FITS_CSV)
    rows = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        d = np.load(full_data_npz(dataset))
        inp = np.asarray(d["inputs"][:, :, sensors, 0])       # (n, 12, S)
        tgt = np.asarray(d["target"][:, :, sensors, 0])       # (n, 12, S)
        n = tgt.shape[0]

        # raw flow for the identifiable sensors, for the 24-step pure-history window
        desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
        raw = np.memmap(REPO / "datasets" / dataset / "data.dat", dtype=np.float32,
                        mode="r", shape=tuple(desc["shape"]))
        flow = np.asarray(raw[:RANGE_END, sensors, 0], np.float32)   # (RANGE_END, S)
        t0 = TEST_OFFSET + np.arange(n) + T_IN - 1                   # forecast origin abs time
        idx = t0[:, None] + np.arange(-23, 1)[None, :]              # (n, 24) history times <= t0
        hist24 = flow[idx]                                          # (n, 24, S)

        hf24 = np.transpose(np.concatenate([inp, tgt], 1), (0, 2, 1))  # (n, S, 24)
        h12 = np.transpose(inp, (0, 2, 1))                             # (n, S, 12)
        h24 = np.transpose(hist24, (0, 2, 1))                          # (n, S, 24)

        om_hf24 = spectral_forecastability(hf24)
        om_h12 = spectral_forecastability(h12)
        om_h24 = spectral_forecastability(h24)

        valid_hf = np.all(hf24 != NULL_VAL, axis=-1)
        valid_h24 = np.all(h24 != NULL_VAL, axis=-1)
        valid = valid_hf & valid_h24                                   # common clean subset

        for c, name in enumerate(ORDER):
            for sl in range(len(sensors)):
                sel = (code[:, sl] == c) & valid[:, sl]
                if sel.sum() < 20:
                    continue
                rows.append({"dataset": dataset, "sensor": int(sensors[sl]),
                             "traffic_state_transition": name, "n": int(sel.sum()),
                             "omega_hf24": float(om_hf24[sel, sl].mean()),
                             "omega_h12": float(om_h12[sel, sl].mean()),
                             "omega_h24": float(om_h24[sel, sl].mean())})
    return pd.DataFrame(rows)


def clustered_ci(df, feat, target="degradation", n_boot=N_BOOT):
    """Detector-clustered bootstrap 95% CI for Spearman(feat, target)."""
    rng = np.random.default_rng(SEED)
    dets = df[["dataset", "sensor"]].drop_duplicates().to_numpy()
    groups = {tuple(k): g for k, g in df.groupby(["dataset", "sensor"])}
    keys = list(groups.keys())
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sub = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        s = sub.dropna(subset=[feat, target])
        if len(s) > 10:
            boot.append(spearmanr(s[feat], s[target])[0])
    boot = np.array(boot)
    point = spearmanr(df[feat], df[target])[0]
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(keys)


def main() -> None:
    df = build()
    df.to_csv(OUT / "fd_foreca_prospective.csv", index=False)
    red = pd.read_csv(OUT / "fd_state_reducibility.csv")
    m = df.merge(red[["dataset", "sensor", "traffic_state_transition",
                      "degradation", "skill", "full_mae"]],
                 on=["dataset", "sensor", "traffic_state_transition"], how="inner")
    thin = m[m.traffic_state_transition != "free_to_free"].dropna(
        subset=["omega_hf24", "omega_h12", "omega_h24", "degradation"])

    print(f"=== degradation Spearman + detector-clustered 95% CI (non-free, n_cells={len(thin)}) ===")
    for feat, desc in [("omega_hf24", "24-step hist+FUTURE (leaky, reproduces 0.475)"),
                       ("omega_h12", "12-step history only"),
                       ("omega_h24", "24-step PURE history (leak-free, length-matched)"),
                       ("skill", "model-based s (ref)")]:
        p, lo, hi, ndet = clustered_ci(thin, feat)
        print(f"  {desc:48s}: {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (clusters={ndet})")

    print("\n=== leakage vs length decomposition ===")
    print(f"  leaky 24 (h+f)      : {spearmanr(thin.omega_hf24, thin.degradation)[0]:+.3f}")
    print(f"  pure-history 24     : {spearmanr(thin.omega_h24, thin.degradation)[0]:+.3f}"
          "   <- same length as leaky, no future => drop from leaky = pure LEAKAGE")
    print(f"  history 12          : {spearmanr(thin.omega_h12, thin.degradation)[0]:+.3f}"
          "   <- h24 vs h12 gap = spectral-LENGTH effect")

    print("\n=== per-dataset (omega_h24) ===")
    for ds in DATASETS:
        s = thin[thin.dataset == ds]
        print(f"  {ds:16s}: {spearmanr(s.omega_h24, s.degradation)[0]:+.3f} (n={len(s)})")

    print("\n=== within-state (omega_h24) + partial|full_mae ===")
    for st in ORDER:
        s = m[m.traffic_state_transition == st].dropna(subset=["omega_h24", "degradation", "full_mae"])
        if len(s) < 8:
            print(f"  {st:24s}   -- ({len(s)})"); continue
        raw = spearmanr(s.omega_h24, s.degradation)[0]
        # partial controlling full_mae, rank residualization
        def rk(x): r = pd.Series(x).rank().to_numpy(); return (r - r.mean())/(r.std()+1e-9)
        Z = np.column_stack([np.ones(len(s)), rk(s.full_mae)])
        rx = rk(s.omega_h24) - Z @ np.linalg.lstsq(Z, rk(s.omega_h24), rcond=None)[0]
        ry = rk(s.degradation) - Z @ np.linalg.lstsq(Z, rk(s.degradation), rcond=None)[0]
        print(f"  {st:24s} raw {raw:+.3f}  partial|full_mae {spearmanr(rx, ry)[0]:+.3f}  (n={len(s)})")
    print("\nwrote fd_foreca_prospective.csv", flush=True)


if __name__ == "__main__":
    main()
