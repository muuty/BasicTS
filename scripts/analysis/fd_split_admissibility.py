#!/usr/bin/env python3
"""Composition of the retained set as the split point moves, at fixed budget.

The split-budget objective spends $r_u$ of the budget on evenly spaced windows and
$r - r_u$ on the windows carrying the most simultaneous breakdown detectors.  The
question the two trained settings do not answer is where $r_u$ should sit.

The composition of the retained set answers it without training.  Write $\\pi_h$ for
the share of state $h$'s atoms the objective retains.  Random selection gives
$\\pi_h = r$ for every state.  A selector that raises breakdown exposure without
moving the rest of the training measure satisfies

    pi_breakdown > r   and   pi_h <= r  for every other state,

and both sides are computed on the chronological training split before any model is
fitted.  This script sweeps the ranked share and reports where that condition holds,
so the split point is read off a measured curve instead of chosen.

Writes experiments/result/analysis/fd_split_admissibility.csv and a figure.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import fd_spatial_synchronisation as sync  # noqa: E402
from select_fd_hybrid import select  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"
FIG = REPO / "writing" / "CoresetSelection-paper" / "figures"
STATES = {0: "free", 1: "breakdown", 2: "recovery", 3: "sustained"}
LABEL = {"free": "Free flow", "breakdown": "Breakdown",
         "recovery": "Recovery", "sustained": "Sustained congestion"}
STYLE = {"free": dict(ls="-", lw=1.4), "breakdown": dict(ls="-", lw=2.4),
         "recovery": dict(ls="--", lw=1.4), "sustained": dict(ls=":", lw=1.6)}


def sweep(dataset: str, ratios: list[float], screen_args) -> pd.DataFrame:
    _, transition, _ = sync.label_matrix(dataset, screen_args)
    n_windows = transition.shape[0]
    counts = {c: (transition == c).sum(axis=1).astype(float) for c in STATES}
    totals = {c: counts[c].sum() for c in STATES}

    rows = []
    for ratio in ratios:
        steps = np.round(np.arange(0.0, ratio + 1e-9, ratio / 10.0), 4)
        for ranked in steps:
            uniform = round(ratio - ranked, 4)
            idx = select(counts[1], n_windows, ratio, uniform)
            row = {"dataset": dataset, "ratio": ratio, "ranked_share": float(ranked),
                   "uniform_share": float(uniform)}
            for code, name in STATES.items():
                row[f"pi_{name}"] = float(counts[code][idx].sum() / totals[code])
            row["admissible"] = bool(
                row["pi_breakdown"] > ratio
                and max(row[f"pi_{n}"] for n in ("free", "recovery", "sustained")) <= ratio)
            rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> None:
    ratios = sorted(df.ratio.unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(3.4 * len(ratios), 3.0),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, ratio in zip(axes, ratios):
        sub = df[df.ratio == ratio].sort_values("ranked_share")
        for name in ("free", "breakdown", "recovery", "sustained"):
            ax.plot(sub.ranked_share, sub[f"pi_{name}"], color="black",
                    label=LABEL[name], **STYLE[name])
        ax.axhline(ratio, color="0.6", lw=0.9)
        ok = sub[sub.admissible]
        if len(ok):
            ax.axvspan(ok.ranked_share.min(), ok.ranked_share.max(),
                       color="0.90", zorder=0)
        ax.set_xlabel("Budget on the ranked part", fontsize=9)
        ax.set_title(f"$r={ratio}$", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Retained share of a state's atoms", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    screen_args = types.SimpleNamespace(
        n_bins=40, min_branch_bins=6, min_free_rise=0.10, min_congested_drop=0.05,
        min_fit_improvement=0.10, min_valid_fraction=2 / 3, min_agreement=0.75)
    frames = [sweep(ds, args.ratios, screen_args) for ds in args.datasets]
    df = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fd_split_admissibility.csv", index=False)

    for (ds, ratio), sub in df.groupby(["dataset", "ratio"]):
        ok = sub[sub.admissible].ranked_share
        span = f"{ok.min():.3f}--{ok.max():.3f}" if len(ok) else "empty"
        print(f"\n=== {ds}  r={ratio}  admissible ranked share: {span} ===")
        print(f"{'ranked':>8}{'free':>9}{'brkdn':>9}{'recov':>9}{'sust':>9}   ok")
        for _, r in sub.sort_values("ranked_share").iterrows():
            print(f"{r.ranked_share:8.3f}{r.pi_free:9.3f}{r.pi_breakdown:9.3f}"
                  f"{r.pi_recovery:9.3f}{r.pi_sustained:9.3f}   {'yes' if r.admissible else ''}")
    plot(df[df.dataset == args.datasets[0]], FIG / "fd_split_admissibility")
    print(f"\nwrote {OUT / 'fd_split_admissibility.csv'} and the figure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["SAN_BERNARDINO", "CONTRA_COSTA"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.3])
    return parser.parse_args()


if __name__ == "__main__":
    main()
