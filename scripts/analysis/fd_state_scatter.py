#!/usr/bin/env python3
"""Reduction loss against the future gap and against the margin over persistence.

The claim these panels carry is a within-versus-between statement: inside a traffic
state the reduction loss is not ordered by the future gap, and the positive
association appears only between the four state means.  A forest of correlation
intervals asserts that; a scatter shows it, because the reader sees the four clouds
and the line through their centres at the same time.

One point is one detector, averaged over the five architectures.  The left panel
plots the future gap, which a selection objective controls and which is computed
before training; the right panel plots the margin over persistence divided by the
persistence error, which is read after training.

Writes fd_state_scatter.pdf and .png into the manuscript figure directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1].parent
OUT = REPO / "experiments" / "result" / "analysis"
FIG = REPO / "writing" / "CoresetSelection-paper" / "figures"

STATES = [
    ("free_to_free", "Free flow", "#2a78d6", "o"),
    ("breakdown", "Breakdown", "#eb6834", "s"),
    ("recovery", "Recovery", "#1baf7a", "^"),
    ("congested_to_congested", "Congestion", "#eda100", "D"),
]


def load(method: str, ratio: float) -> pd.DataFrame:
    panel = pd.read_csv(OUT / "fd_state_margin_panel.csv")
    audit = pd.read_csv(OUT / "fd_sensor_resolved_seed_averaged.csv")
    gap = (audit[(audit.method == method) & (audit.ratio == ratio) & (audit.K_label == "K0")]
           .groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
           ["future_gap"].mean())
    df = panel.merge(gap, on=["dataset", "sensor", "traffic_state_transition"], how="inner")
    df["margin_ratio"] = (df.persist_mae - df.full_mae) / df.persist_mae
    return (df.groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
            .agg(loss=("degradation", "mean"), gap=("future_gap", "mean"),
                 margin_ratio=("margin_ratio", "mean")))


def panel(ax, df, xcol, xlabel):
    centres = []
    for code, label, colour, marker in STATES:
        sub = df[df.traffic_state_transition == code].dropna(subset=[xcol, "loss"])
        if sub.empty:
            continue
        ax.scatter(sub[xcol], sub.loss, s=9, marker=marker, facecolor=colour,
                   edgecolor="none", alpha=0.30, zorder=2)
        r = spearmanr(sub[xcol], sub.loss).statistic
        cx, cy = sub[xcol].median(), sub.loss.median()
        centres.append((cx, cy, colour, marker, label, r, len(sub)))
    centres.sort(key=lambda t: t[0])
    ax.plot([c[0] for c in centres], [c[1] for c in centres], color="0.35",
            lw=1.1, ls="--", zorder=3)
    for cx, cy, colour, marker, label, r, n in centres:
        ax.plot([cx], [cy], marker=marker, ms=10, mfc=colour, mec="black",
                mew=1.1, ls="none", zorder=4, label=f"{label}  $\\rho={r:+.2f}$")
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    return centres


def main() -> None:
    args = parse_args()
    df = load(args.method, args.ratio)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    left = panel(axes[0], df, "gap", "Future gap $g$ (before training)")
    panel(axes[1], df, "margin_ratio", "Margin over persistence $M/e_p$ (after training)")
    axes[0].set_ylabel("Reduction loss $\\Delta$ (MAE)", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right", handletextpad=0.3)
    axes[0].set_ylim(np.nanpercentile(df.loss, 0.5), np.nanpercentile(df.loss, 99.5))
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fd_state_scatter.{ext}", dpi=300, bbox_inches="tight")

    print("within-state Spearman (one point = one detector, architectures averaged):")
    for cx, cy, _, _, label, r, n in sorted(left, key=lambda t: t[4]):
        print(f"  {label:11s} n={n:4d}  rho(g, loss) = {r:+.3f}   "
              f"median g {cx:.3f}  median loss {cy:.2f}")
    cs = sorted(left, key=lambda t: t[0])
    print(f"\nbetween the four state medians: Pearson "
          f"{np.corrcoef([c[0] for c in cs], [c[1] for c in cs])[0, 1]:+.3f}")
    print(f"\nwrote {FIG / 'fd_state_scatter.{pdf,png}'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="k_medoids")
    parser.add_argument("--ratio", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
