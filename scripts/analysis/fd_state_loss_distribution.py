#!/usr/bin/env python3
"""Distribution of the per-detector reduction loss, by traffic state and objective.

The state-conditional table reports a mean per state.  A mean does not say whether
reduction costs a little at every detector or a great deal at a few, and it does not
show what the worst detectors lose.  This script plots a kernel density estimate of
the per-detector loss for the four traffic states on one pair of axes, once for the
random baseline and once for the split-budget objective, so the four states can be
compared directly and the effect of the objective is visible as a horizontal shift.

Architectures are averaged within detector, over the architectures both objectives
were trained on, and detectors with fewer than --min-obs masked observations in a
state are dropped: at breakdown the thinnest detectors carry a handful of atoms and
their loss is measurement noise.

Reads fd_selection_experiment_by_detector.csv; writes fd_state_loss_distribution.pdf
and .png into the manuscript figure directory, and prints the quantiles it draws.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1].parent
OUT = REPO / "experiments" / "result" / "analysis"
FIG = REPO / "writing" / "CoresetSelection-paper" / "figures"

# Categorical slots 1-4 of the validated palette, assigned in the canonical state
# order and never cycled.  The lower-contrast slots carry a line style as secondary
# encoding, so identity never rests on colour alone.
STATES = [
    ("free", "Free flow", "#2a78d6", "-"),
    ("breakdown", "Breakdown", "#eb6834", "-"),
    ("recovery", "Recovery", "#1baf7a", "--"),
    ("sustained", "Congestion", "#eda100", ":"),
]
PANELS = [("random", "Random"), ("fd_hyb0604", "Split $0.06+0.04$")]


def per_detector_loss(df: pd.DataFrame, methods: list[str], min_obs: int) -> pd.DataFrame:
    """Architecture-averaged per-detector loss against the full-data run."""
    full = (df[df.method == "full"]
            .rename(columns={"mae": "mae_full"})
            [["dataset", "backbone", "sensor", "state", "mae_full"]])
    red = df[df.method.isin(methods)].merge(
        full, on=["dataset", "backbone", "sensor", "state"], how="inner")
    red = red[red.n_obs >= min_obs]
    red["loss"] = red["mae"] - red["mae_full"]
    common = None
    for m in methods:
        got = set(red[red.method == m].backbone)
        common = got if common is None else (common & got)
    red = red[red.backbone.isin(common)]
    return (red.groupby(["dataset", "sensor", "state", "method"], as_index=False)
            .agg(loss=("loss", "mean")))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(OUT / "fd_selection_experiment_by_detector.csv")
    df = df[df.dataset == args.dataset]
    loss = per_detector_loss(df, [p[0] for p in PANELS], args.min_obs)

    lo = float(np.nanpercentile(loss.loss, 1.0))
    hi = float(np.nanpercentile(loss.loss, 99.0))
    grid = np.linspace(lo, hi, 512)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    ymax = 0.0
    for ax, (method, panel) in zip(axes, PANELS):
        for state, label, colour, dash in STATES:
            v = loss[(loss.state == state) & (loss.method == method)].loss.to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size < 5:
                continue
            dens = gaussian_kde(v)(grid)
            ymax = max(ymax, dens.max())
            ax.fill_between(grid, dens, color=colour, alpha=0.13, lw=0)
            ax.plot(grid, dens, color=colour, lw=1.7, ls=dash, label=label)
            print(f"{panel:18s}{label:11s} n={v.size:4d}  median {np.median(v):6.2f}  "
                  f"p90 {np.percentile(v, 90):6.2f}")
        ax.axvline(0.0, color="0.65", lw=0.8, zorder=0)
        ax.set_title(panel, fontsize=10)
        ax.set_xlabel("Per-detector reduction loss (MAE)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Density over detectors", fontsize=9)
    axes[0].set_ylim(0, ymax * 1.12)
    axes[0].set_xlim(lo, hi)
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fd_state_loss_distribution.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nwrote {FIG / 'fd_state_loss_distribution.{pdf,png}'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SAN_BERNARDINO")
    parser.add_argument("--min-obs", type=int, default=120,
                        help="Minimum masked observations per detector and state.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
