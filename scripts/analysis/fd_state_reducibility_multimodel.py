#!/usr/bin/env python3
"""Per-backbone reducibility and its association with reduced-data degradation.

Extends ``fd_state_conditional_followups.reducibility`` (which averages degradation
over backbones and takes reducibility from STGCN alone) to a per-backbone panel, so
the manuscript can state the result across all five forecasting architectures rather
than for STGCN only.

Conventions are inherited from that module so the numbers stay comparable:
  * support difficulty D  : fd_sensor_resolved_by_detector.csv, method/ratio/K_label
                            filtered, averaged over selection seeds
  * full / reduced MAE    : fd_state_conditional_mae_by_detector.csv, per backbone
  * persistence MAE       : last-value forecast from the full-data STGCN run's inputs
                            (model-free, so it is shared across backbones)

Reported per (dataset, sensor, traffic_state, backbone):
  reducibility = 1 - full_mae / persist_mae      (error reduction over persistence)
  degradation  = red_mae - full_mae

Outputs
  fd_state_reducibility_multimodel.csv  - the per-detector panel
  fd_state_reducibility_summary.csv     - the within-state association summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_state_conditional_followups import (  # noqa: E402
    OUT, PHASE_C, FITS_CSV, AUDIT_CSV, MAE_DETECTOR_CSV, DATASETS, BACKBONE_DIR,
    ORDER, test_code, detector_state_mae_perdet,
)

BACKBONES = ["STGCN", "STID", "DCRNN", "AGCRN", "STAEformer"]
KEY = ["dataset", "sensor", "traffic_state_transition"]


def persistence_per_detector(fits: pd.DataFrame) -> pd.DataFrame:
    """Model-free last-value MAE per (dataset, sensor, state), from the STGCN run."""
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
        persist = np.repeat(inp[:, -1:, :], tgt.shape[1], axis=1)
        f = detector_state_mae_perdet(persist, tgt, code, sensors)
        f = f.rename(columns={"mae": "persist_mae"})
        f["dataset"] = dataset
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def panel(fits, method="k_medoids", ratio=0.3, k_label="K50") -> pd.DataFrame:
    audit = pd.read_csv(AUDIT_CSV)
    audit = audit[(audit.method == method) & (audit.ratio == ratio) & (audit.K_label == k_label)]
    audit = audit.groupby(KEY, as_index=False)["D"].mean()

    mae = pd.read_csv(MAE_DETECTOR_CSV)
    full = (mae[mae.method == "full"]
            .groupby(KEY + ["backbone"], as_index=False)["mae"].mean()
            .rename(columns={"mae": "full_mae"}))
    red = (mae[(mae.method == method) & (mae.ratio == ratio)]
           .groupby(KEY + ["backbone"], as_index=False)["mae"].mean()
           .rename(columns={"mae": "red_mae"}))

    df = (red.merge(full, on=KEY + ["backbone"])
             .merge(audit, on=KEY)
             .merge(persistence_per_detector(fits), on=KEY))
    df["degradation"] = df.red_mae - df.full_mae
    df["reducibility"] = 1 - df.full_mae / df.persist_mae
    return df[np.isfinite(df.reducibility) & (df.persist_mae > 0)]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def pcorr(a, b, c):
    rab, rac, rbc = corr(a, b), corr(a, c), corr(b, c)
    den = ((1 - rac ** 2) * (1 - rbc ** 2)) ** 0.5
    return (rab - rac * rbc) / den if den > 0 else np.nan


def main() -> None:
    fits = pd.read_csv(FITS_CSV)
    df = panel(fits)
    df.to_csv(OUT / "fd_state_reducibility_multimodel.csv", index=False)

    rows = []
    for b in BACKBONES:
        for s in ORDER:
            x = df[(df.backbone == b) & (df.traffic_state_transition == s)]
            rows.append(dict(
                backbone=b, traffic_state_transition=s, n_detectors=len(x),
                reducibility_mean=x.reducibility.mean(),
                degradation_mean=x.degradation.mean(),
                r_reducibility_degradation=corr(x.reducibility, x.degradation),
                partial_r_given_full_mae=pcorr(x.reducibility, x.degradation, x.full_mae),
                r_D_degradation=corr(x.D, x.degradation),
            ))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "fd_state_reducibility_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== reducibility by backbone x state (mean over detectors) ===")
    print(summary.pivot(index="backbone", columns="traffic_state_transition",
                        values="reducibility_mean").loc[BACKBONES, ORDER].round(3))
    print("\n=== within-state r(reducibility, degradation) ===")
    print(summary.pivot(index="backbone", columns="traffic_state_transition",
                        values="r_reducibility_degradation").loc[BACKBONES, ORDER].round(3))
    print("\n=== within-state partial r, controlling full MAE ===")
    p = summary.pivot(index="backbone", columns="traffic_state_transition",
                      values="partial_r_given_full_mae").loc[BACKBONES, ORDER]
    print(p.round(3))
    v = summary.partial_r_given_full_mae
    print(f"\n{len(summary)} cells: all positive = {bool((v > 0).all())}; "
          f"min = {v.min():.3f}; median = {v.median():.3f}")

    print("\n=== coverage decoupling: within-state r(D, degradation) ===")
    print(summary.pivot(index="backbone", columns="traffic_state_transition",
                        values="r_D_degradation").loc[BACKBONES, ORDER].round(3))
    print(f"\npooled r(D, degradation) = {corr(df.D, df.degradation):+.3f}  (n={len(df)})")
    print(f"pooled r(reducibility, degradation) = {corr(df.reducibility, df.degradation):+.3f}")
    print("\nwrote fd_state_reducibility_{multimodel,summary}.csv")


if __name__ == "__main__":
    main()
