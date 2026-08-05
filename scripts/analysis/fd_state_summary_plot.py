#!/usr/bin/env python3
"""State-conditional summary: coverage difficulty, reduction loss, and reducible
margin for the four fundamental-diagram states, with detector-bootstrap CIs.

Companion to fd_state_examples_plot.py. Reads fd_state_margin_by_state.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
PAPER = REPO / "writing" / "CoresetSelection-paper" / "figures"

ORDER = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
LABELS = ["Free flow", "Breakdown", "Recovery", "Congestion"]
BAR = "#555555"

# (column, ci_low, ci_high, panel title, y-label)
PANELS = [
    ("D", "D_ci_low", "D_ci_high", "Coverage difficulty", r"$D$"),
    ("full_gain", "full_gain_ci_low", "full_gain_ci_high",
     "Reducibility", r"reducible margin $M$ (MAE)"),
    ("gain_lost", "gain_lost_ci_low", "gain_lost_ci_high",
     "MAE reduction loss", r"$\Delta$ (MAE)"),
]


def main():
    df = pd.read_csv(OUT / "fd_state_margin_by_state.csv").set_index(
        "traffic_state_transition"
    ).loc[ORDER]
    x = np.arange(len(ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (col, lo, hi, title, ylab) in zip(axes, PANELS):
        val = df[col].to_numpy()
        err = np.vstack([val - df[lo].to_numpy(), df[hi].to_numpy() - val])
        ax.bar(x, val, color=BAR, width=0.66, zorder=2)
        ax.errorbar(x, val, yerr=err, fmt="none", ecolor="0.15",
                    elinewidth=1.1, capsize=3, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.margins(y=0.12)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    for d in (OUT, PAPER):
        for ext in ("png", "pdf"):
            fig.savefig(d / f"fd_state_summary.{ext}", dpi=200, bbox_inches="tight")
    print("wrote fd_state_summary.{png,pdf} to analysis/ and paper figures/")
    for lab, s in zip(LABELS, ORDER):
        print(f"  {lab:12} D={df.loc[s,'D']:.3f}  M={df.loc[s,'full_gain']:.2f}  "
              f"Delta={df.loc[s,'gain_lost']:.2f}")


if __name__ == "__main__":
    main()
