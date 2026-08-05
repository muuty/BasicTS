#!/usr/bin/env python3
"""The attainable retention frontier, and what training on each point recovers.

The frontier of the retention proposition is a curve in the plane of two retention
rates: how much free flow a budget holds, and how much breakdown it can then reach.
Each audited objective is one point in that plane, and the colour is what training on
it actually costs at breakdown. Reading position alone orders nothing; reading both
coordinates does, which is the point of the figure.

Colour is sequential because the encoded quantity is a magnitude: one hue, five steps,
relative luminance monotone from 0.807 to 0.104. Every point is directly labelled,
since the marks sit below the 3:1 contrast line against the surface.

A second figure carries the same data against the share of the bound attained, with
the three state losses as separate series, because the paper's claim is about what
approaching the bound costs and that is not readable from the plane.

Writes figures/fd_frontier.{png,pdf} and figures/fd_frontier_loss.{png,pdf}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

REPO = Path(__file__).resolve().parents[1].parent
OUT = REPO / "experiments" / "result" / "analysis"
FIG = REPO / "writing" / "CoresetSelection-paper" / "figures"

RAMP = ["#fde3d6", "#f9b696", "#f18659", "#d95c25", "#9c3d12"]
CMAP = LinearSegmentedColormap.from_list("loss", RAMP)
INK, MUTED = "#1a1a1a", "#8a8a8a"

MODEL_FREE = ["random", "stride", "recent", "k_medoids", "k_center", "graph_cut"]
DESIGNED = {"fd_grp00": r"Quota $\beta{=}0$", "fd_hyb0802": r"Split $.08{+}.02$",
            "fd_hyb0604": r"Split $.06{+}.04$", "fd_strat00": "Stratified allocation",
            "fd_front10": "Frontier set"}
# Offsets in points, per panel, set after rendering so no label crosses a mark.
NUDGE = {"fd_grp00": (10, -3, "left"), "fd_hyb0802": (-10, -3, "right"),
         "fd_hyb0604": (-10, -3, "right"), "fd_strat00": (0, 11, "center"),
         "fd_front10": (10, 3, "left")}
PANELS = [("SAN_BERNARDINO", "San Bernardino"), ("CONTRA_COSTA", "Contra Costa")]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = pd.read_csv(OUT / "attainable_allocation.csv")
    curve = curve[(curve.target == "breakdown") & (curve.guarded == "free")
                  & (curve.ratio == 0.1)].dropna(subset=["lp"])
    points = pd.read_csv(OUT / "attainable_allocation_selectors.csv")
    points = points[(points.target == "breakdown") & (points.guarded == "free")
                    & (points.ratio == 0.1)]
    loss = pd.read_csv(OUT / "fd_selection_experiment_summary.csv")
    loss = (loss[(loss.ratio == 0.1) & (loss.state == "breakdown")]
            .groupby(["dataset", "method"], as_index=False).degradation.mean()
            .rename(columns={"method": "objective"}))
    return curve, points.merge(loss, on=["dataset", "objective"])


def main() -> None:
    parse_args()
    curve, points = load()
    norm = Normalize(vmin=points.degradation.min(), vmax=points.degradation.max())

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        c = curve[curve.dataset == key].sort_values("tau")
        unconstrained = float(c.lp.max())
        binding = c[np.isclose(c.guarded_at_attainable, c.tau, atol=2e-3)]
        corner = float(c.guarded_at_attainable.min())

        # No set at this budget exceeds the unconstrained maximum, at any guard level.
        ax.axhline(unconstrained, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(0.003, unconstrained + 0.012, "unconstrained maximum",
                fontsize=7, color=MUTED)
        # Where the guard binds, the boundary falls away.
        ax.plot([corner] + list(binding.guarded_at_attainable),
                [unconstrained] + list(binding.lp),
                color=INK, lw=2, zorder=2, solid_capstyle="round")
        ax.text(0.146, 0.185, "attainable\nboundary", fontsize=7.5, color=INK,
                ha="center", va="center")

        p = points[points.dataset == key]
        free = p[p.objective.isin(MODEL_FREE)]
        ax.scatter(free.pi_guarded, free.pi_target, s=52, marker="o",
                   c=CMAP(norm(free.degradation)), edgecolor="white", linewidth=1.4,
                   zorder=3)
        ax.annotate("six model-free\nobjectives", (0.060, 0.048), fontsize=7.5,
                    color=INK, ha="center", va="center")

        made = p[p.objective.isin(DESIGNED)]
        ax.scatter(made.pi_guarded, made.pi_target, s=86, marker="D",
                   c=CMAP(norm(made.degradation)), edgecolor="white", linewidth=1.6,
                   zorder=4)
        for _, r in made.iterrows():
            dx, dy, align = NUDGE[r.objective]
            ax.annotate(DESIGNED[r.objective], (r.pi_guarded, r.pi_target),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.5,
                        color=INK, zorder=5, ha=align)

        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel(r"Free-flow retention $\pi_{\mathrm{free}}$", fontsize=9)
        ax.set_xlim(0.0, 0.17)
        ax.set_ylim(0.0, 0.70)
        ax.grid(True, lw=0.5, color="#ececec", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
    axes[0].set_ylabel(r"Breakdown retention $\pi_{\mathrm{bd}}$", fontsize=9)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
                       fraction=0.032, pad=0.02)
    bar.set_label("Breakdown reduction loss (MAE)", fontsize=8, color=INK)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.outline.set_visible(False)

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fd_frontier.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {FIG / 'fd_frontier.png'}")
    loss_figure()


SERIES = [("breakdown", "Breakdown", "#eb6834", "o", "-"),
          ("recovery", "Recovery", "#1baf7a", "s", "--"),
          ("free", "Free flow", "#2a78d6", "^", ":")]
# Objectives that hold free-flow retention within a hundredth of the nominal rate lie on
# one path; the two that pay for breakdown retention out of free flow do not, so their
# points are shown unconnected. Joining all nine would draw a trajectory nobody travels.
HELD = ["random", "fd_hyb0802", "fd_hyb0604", "fd_front10"]
BROKEN = {"fd_grp00": r"quota", "fd_strat00": "stratified", "graph_cut": "Graph Cut"}


def loss_figure() -> None:
    """What the loss does as an objective approaches the bound at held composition."""
    points = pd.read_csv(OUT / "attainable_allocation_selectors.csv")
    points = points[(points.target == "breakdown") & (points.guarded == "free")
                    & (points.ratio == 0.1)]
    loss = pd.read_csv(OUT / "fd_selection_experiment_summary.csv")
    loss = loss[loss.ratio == 0.1].pivot_table(
        index=["dataset", "method"], columns="state", values="degradation").reset_index()
    d = points.merge(loss.rename(columns={"method": "objective"}), on=["dataset", "objective"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        g = d[d.dataset == key]
        path = g[g.objective.isin(HELD)].sort_values("attained_share")
        off = g[g.objective.isin(BROKEN)]
        for col, label, colour, marker, style in SERIES:
            ax.plot(path.attained_share, path[col], color=colour, lw=2, ls=style,
                    marker=marker, ms=6.5, mec="white", mew=1.3, label=label, zorder=3)
            ax.scatter(off.attained_share, off[col], s=34, facecolor="white",
                       edgecolor=colour, linewidth=1.4, zorder=2)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("Share of the attainable bound reached", fontsize=9)
        ax.set_xlim(0.03, 1.07)
        ax.set_ylim(0, 40)
        ax.grid(True, lw=0.5, color="#ececec", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
    axes[0].set_ylabel("Reduction loss (MAE)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center", ncol=1)
    axes[1].text(1.04, 24, "hollow: free-flow\nretention given up",
                 fontsize=7.5, color=MUTED, ha="right", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fd_frontier_loss.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {FIG / 'fd_frontier_loss.png'}")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    main()
