#!/usr/bin/env python3
"""Correlations of the reduction loss with the future gap, replacing the assignment gap.

The manuscript reported the association between the reduction loss and a calibrated
combination of the history distance and the future gap.  That combination tracked the
future gap alone at a per-detector rank correlation above 0.999 in three of the four
states, so it carried no information the future gap does not, and its calibration
constant was one more quantity for the reader to hold.  This script recomputes every
statistic the manuscript quoted for it on the future gap $g_a$ of Equation
(future_gap), with the same detector bootstrap.

Writes experiments/result/analysis/fd_future_gap_correlations.csv with the column
names fd_forest_plot.py expects, so the forest figure can be regenerated against it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1].parent
OUT = REPO / "experiments" / "result" / "analysis"
PANEL = OUT / "fd_state_margin_panel.csv"
AUDIT = OUT / "fd_sensor_resolved_seed_averaged.csv"

STATES = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
LABEL = {"free_to_free": "free", "breakdown": "breakdown",
         "recovery": "recovery", "congested_to_congested": "sustained"}


def bootstrap_r(x: np.ndarray, y: np.ndarray, reps: int, rng) -> tuple[float, float, float]:
    """Spearman correlation with a percentile detector bootstrap."""
    point = spearmanr(x, y).statistic
    n = len(x)
    draws = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            draws[b] = np.nan
            continue
        draws[b] = spearmanr(x[idx], y[idx]).statistic
    draws = draws[np.isfinite(draws)]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    panel = pd.read_csv(PANEL)
    audit = pd.read_csv(AUDIT)
    gap = (audit[(audit.method == args.method) & (audit.ratio == args.ratio)
                 & (audit.K_label == "K0")]
           .groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
           ["future_gap"].mean())
    df = panel.merge(gap, on=["dataset", "sensor", "traffic_state_transition"], how="inner")

    rows = []
    for state in STATES:
        for backbone, cell in df[df.traffic_state_transition == state].groupby("backbone"):
            cell = cell.dropna(subset=["future_gap", "degradation"])
            r, lo, hi = bootstrap_r(cell.future_gap.to_numpy(float),
                                    cell.degradation.to_numpy(float), args.reps, rng)
            rows.append({"backbone": backbone, "traffic_state_transition": state,
                         "n_detectors": len(cell), "r_g_degradation": r,
                         "r_g_degradation_ci_low": lo, "r_g_degradation_ci_high": hi})
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "fd_future_gap_correlations.csv", index=False)

    print(f"=== r(g, degradation), {args.method} r={args.ratio}, "
          f"{args.reps} detector bootstrap replicates ===")
    below = int((corr.r_g_degradation_ci_high < 0).sum())
    above = int((corr.r_g_degradation_ci_low > 0).sum())
    print(f"{len(corr)} architecture--state cells: {below} intervals entirely below zero, "
          f"{above} entirely above; median r = {corr.r_g_degradation.median():+.3f}")
    for state in STATES:
        sub = corr[corr.traffic_state_transition == state]
        print(f"  {LABEL[state]:10s} median {sub.r_g_degradation.median():+.3f}  "
              f"below-zero {int((sub.r_g_degradation_ci_high < 0).sum())}/{len(sub)}")

    # Architecture-averaged detector values, the form the text quotes.
    av = (df.groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
          .agg(degradation=("degradation", "mean"), g=("future_gap", "mean")))
    print("\n=== architecture-averaged detector values ===")
    means = []
    for state in STATES:
        cell = av[av.traffic_state_transition == state].dropna()
        r = spearmanr(cell.g, cell.degradation).statistic
        print(f"  {LABEL[state]:10s} Spearman {r:+.3f} on {len(cell)} detectors, "
              f"mean g = {cell.g.mean():.3f}, mean degradation = {cell.degradation.mean():.3f}")
        means.append((cell.g.mean(), cell.degradation.mean()))
    g_means = np.array([m[0] for m in means])
    d_means = np.array([m[1] for m in means])
    print(f"  between the four state means: Pearson {np.corrcoef(g_means, d_means)[0, 1]:+.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="k_medoids")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
