#!/usr/bin/env python3
"""Quantization error against test MAE for the six objectives, at three budgets.

Section 6.3 compares selection objectives inside a fixed architecture, network and
budget. This figure shows one such comparison directly: each panel holds the six
objectives at one budget, positioned by the quantization error computed before
training and by the test MAE measured after it.

The three budgets are chosen to show what changes. At r=0.1 the objectives span
almost nine MAE and the two that lose the most sit far out on both axes. At r=0.9
they span a seventh of one MAE, so there is almost nothing left for any diagnostic
to order.

Correlations are reported on the values rather than on the ranks: with six points a
rank correlation discards the magnitudes, which is where the separation is.

Writes fd_quantization_scatter.pdf and .png into the manuscript figure directory, and
fd_quantization_within_block.csv beside the other analysis artefacts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import proxy_heldout_validation as ph  # noqa: E402

FIG = REPO / "writing" / "CoresetSelection-paper" / "figures"
OUT = REPO / "experiments" / "result" / "analysis"
# One colour and six marker shapes: identity never rests on colour, and the figure
# survives greyscale printing.
MARKS = [("k_medoids", "K-medoids", "o"), ("stride", "Stride", "s"),
         ("random", "Random", "^"), ("recent", "Recent", "D"),
         ("k_center", "K-center", "v"), ("graph_cut", "Graph Cut", "P")]
COLOUR = "#2a78d6"


def cells() -> pd.DataFrame:
    merged = ph.load_results().merge(
        ph.load_proxies(), on=["dataset", "method", "ratio", "seed"], how="inner")
    merged = merged.dropna(subset=["quantization_cost", "MAE_mean"])
    return (merged.groupby(["model", "dataset", "ratio", "method"], as_index=False)
            .agg(J=("quantization_cost", "mean"), mae=("MAE_mean", "mean")))




def main() -> None:
    args = parse_args()
    c = cells()
    sub = c[(c.model == args.model) & (c.dataset == args.dataset)]

    fig, axes = plt.subplots(1, len(args.ratios), figsize=(2.55 * len(args.ratios), 2.9))
    axes = np.atleast_1d(axes)
    for ax, ratio in zip(axes, args.ratios):
        g = sub[np.isclose(sub.ratio, ratio)].sort_values("J")
        r = pearsonr(g.J, g.mae)
        for key, label, marker in MARKS:
            cell = g[g.method == key]
            if cell.empty:
                continue
            ax.scatter(cell.J, cell.mae, s=34, marker=marker, facecolor=COLOUR,
                       edgecolor="0.2", linewidth=0.5, zorder=3,
                       label=label if ax is axes[0] else None)
        ax.set_title(f"$r={ratio}$   $r_{{\\mathrm{{P}}}}={r.statistic:+.2f}$", fontsize=9)
        ax.set_xlabel("Quantization error $J_\\psi$", fontsize=8.5)
        ax.margins(x=0.16, y=0.16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
        print(f"r={ratio}: Pearson {r.statistic:+.3f}  Spearman "
              f"{spearmanr(g.J, g.mae).statistic:+.3f}  "
              f"MAE spread {g.mae.max() - g.mae.min():.3f}")
    axes[0].set_ylabel("Test MAE", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, fontsize=7.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.06), handletextpad=0.2,
               columnspacing=1.1)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fd_quantization_scatter.{ext}", dpi=300, bbox_inches="tight")

    rows = []
    for (model, dataset, ratio), g in c.groupby(["model", "dataset", "ratio"]):
        if len(g) >= 4:
            rows.append({"model": model, "dataset": dataset, "ratio": ratio,
                         "n_objectives": len(g),
                         "pearson": pearsonr(g.J, g.mae).statistic,
                         "spearman": spearmanr(g.J, g.mae).statistic,
                         "mae_spread": g.mae.max() - g.mae.min()})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "fd_quantization_within_block.csv", index=False)
    print(f"\nmedian over the ten cells at each budget ({args.model} shown above):")
    print(tab.groupby("ratio")[["pearson", "spearman", "mae_spread"]].median().round(3).to_string())
    print(f"\nwrote {FIG / 'fd_quantization_scatter.{pdf,png}'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="STGCNChebGraphConv")
    parser.add_argument("--dataset", default="SAN_BERNARDINO")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    return parser.parse_args()


if __name__ == "__main__":
    main()
