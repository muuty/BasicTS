#!/usr/bin/env python3
"""Forest plot of the null and its positive control, over the 20 architecture
by traffic-state cells.

Each row carries two correlations with the same outcome, the per-detector MAE
reduction loss Delta:

  filled  r(g, Delta)        -- the future gap, computed from data geometry
                                before any training
  open    r(M/e_p, Delta)    -- the post-hoc skill of full-data training over a
                                last-value forecast, read after training

Both are dataset-adjusted Pearson correlations over the 206-211 screened
detectors of a cell, with 95% intervals from a dataset-stratified detector
bootstrap, as stored by fd_state_margin_reanalysis.py in
fd_state_margin_correlations.csv. Plotting the two together is the point: the
positive control fires in the same cells where the pre-training quantity does
not, so a null on g cannot be read as a broken instrument.

Usage:  conda activate cuda && python scripts/analysis/fd_forest_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
PAPER = REPO / "writing" / "CoresetSelection-paper" / "figures"

STATES = [
    ("free_to_free", "Free flow"),
    ("breakdown", "Breakdown"),
    ("recovery", "Recovery"),
    ("congested_to_congested", "Congestion"),
]
ARCHITECTURES = ["STGCN", "AGCRN", "DCRNN", "STID", "STAEformer"]

SERIES = [
    ("r_g_degradation", "future gap $g$", dict(
        marker="o", ms=4.6, mfc="black", mec="black", mew=1.0)),
    ("r_reducibility_degradation", "margin ratio $M/e_p$", dict(
        marker="s", ms=4.6, mfc="white", mec="black", mew=1.0)),
]
OFFSET = 0.19


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "fd_state_margin_correlations.csv")
    gap = pd.read_csv(OUT / "fd_future_gap_correlations.csv")
    df = df.merge(gap[["backbone", "traffic_state_transition", "r_g_degradation",
                       "r_g_degradation_ci_low", "r_g_degradation_ci_high"]],
                  on=["backbone", "traffic_state_transition"], how="left")
    missing = set(ARCHITECTURES) - set(df.backbone)
    if missing:
        raise AssertionError(f"architectures missing from the table: {missing}")
    return df.set_index(["traffic_state_transition", "backbone"])


def audit(df: pd.DataFrame) -> None:
    """Print the counts the caption states, from the same rows that are drawn."""
    for column, label, _ in SERIES:
        lo = df[f"{column}_ci_low"].to_numpy(float)
        hi = df[f"{column}_ci_high"].to_numpy(float)
        point = df[column].to_numpy(float)
        print(f"{label}: {int((hi < 0).sum())} of {len(point)} intervals entirely "
              f"below zero, {int((lo > 0).sum())} entirely above; "
              f"median r = {np.median(point):+.3f}, "
              f"range [{point.min():+.3f}, {point.max():+.3f}]")


def plot(df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })
    n_rows = len(STATES) * len(ARCHITECTURES)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    y_positions, y_labels, group_spans = [], [], []
    y = n_rows + len(STATES) - 1
    for state, nice in STATES:
        top = y
        for architecture in ARCHITECTURES:
            row = df.loc[(state, architecture)]
            for column, _, style in SERIES:
                sign = -1 if column.startswith("r_g") else 1
                yy = y + sign * OFFSET
                point = float(row[column])
                lo = float(row[f"{column}_ci_low"])
                hi = float(row[f"{column}_ci_high"])
                ax.plot([lo, hi], [yy, yy], color="0.25", lw=0.9, zorder=2,
                        solid_capstyle="butt")
                ax.plot([lo, lo], [yy - 0.12, yy + 0.12], color="0.25", lw=0.9)
                ax.plot([hi, hi], [yy - 0.12, yy + 0.12], color="0.25", lw=0.9)
                ax.plot([point], [yy], ls="none", zorder=3, **style)
            y_positions.append(y)
            y_labels.append(architecture)
            y -= 1
        group_spans.append((nice, top, y + 1))
        y -= 1

    ax.axvline(0.0, color="black", lw=0.9, zorder=1)
    for _, top, bottom in group_spans[1::2]:
        ax.axhspan(bottom - 0.5, top + 0.5, color="0.94", zorder=0)
    blended = matplotlib.transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    )
    for nice, top, bottom in group_spans:
        ax.text(-0.195, (top + bottom) / 2, nice, fontsize=8.5, rotation=90,
                va="center", ha="center", linespacing=1.15, transform=blended)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(y + 0.6, n_rows + len(STATES) - 0.4)
    ax.set_xlim(-0.72, 1.02)
    ax.set_xticks([-0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    ax.set_xlabel(r"correlation with the reduction loss $\Delta$, "
                  "95% detector bootstrap")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], ls="none", label=label, **style)
               for _, label, style in SERIES]
    ax.legend(handles=handles, loc="lower center", frameon=False,
              handlelength=1.0, numpoints=1, ncol=2, columnspacing=2.4,
              borderpad=0.2, bbox_to_anchor=(0.5, 1.005))

    fig.tight_layout()
    PAPER.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"fd_forest.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {PAPER / 'fd_forest.png'} and .pdf")


def main() -> None:
    df = load()
    audit(df)
    plot(df)
    print("\nplotted cells (state / architecture: r(D,Delta) [ci], "
          "r(skill,Delta) [ci]):")
    for state, nice in STATES:
        for architecture in ARCHITECTURES:
            r = df.loc[(state, architecture)]
            flat = nice.replace("\n", " ")
            print(f"  {flat:<21s} {architecture:<11s} n={int(r.n_detectors):3d}  "
                  f"{r.r_D_degradation:+.3f} "
                  f"[{r.r_D_degradation_ci_low:+.3f},"
                  f"{r.r_D_degradation_ci_high:+.3f}]   "
                  f"{r.r_reducibility_degradation:+.3f} "
                  f"[{r.r_reducibility_degradation_ci_low:+.3f},"
                  f"{r.r_reducibility_degradation_ci_high:+.3f}]")


if __name__ == "__main__":
    main()
