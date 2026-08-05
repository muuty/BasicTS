#!/usr/bin/env python3
"""Pre-training scores grouped by what they measure, against realised MAE.

Ten scores were reported earlier as ten competing proxies, which hid two things.
Four of them are the same quantity under different summaries: the mean, median and
maximum distance from a training window to its nearest retained one, and the
facility-location objective, which is that distance passed through a similarity
kernel and summed. And they were not computed in the same space, the coverage scores
on a 50-component projection and the rest on raw standardised trajectories, so any
gap between families measured the representation as much as the property.

compute_proxy_families_raw.py removes the second problem by recomputing every family
from one distance matrix in the raw space Proposition 1 attaches to. This script
groups what it produces by the property measured and scores each against test MAE.

The correlation is Spearman inside a fixed architecture, network and budget, over the
six seed-averaged objectives that cell contains. It is reported per budget and not
pooled, for two reasons. Pooling budgets moves both the score and the error, and one
pooled number per family is decided by which cells enter it: the median
nearest-distance is zero for every objective once most windows are retained, so
requiring all scores to vary restricts the comparison to the three smallest budgets,
which is enough to reverse the family ranking.

Writes tables/proxy_families.tex into the manuscript directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
RESULT = REPO / "experiments" / "result"
TABLE = REPO / "writing" / "CoresetSelection-paper" / "tables" / "proxy_families.tex"

# score -> (family, description). Orientation is not encoded: the table reports the
# absolute correlation, since the question is which property tracks accuracy at all.
FAMILY = {
    "qc_mean":      ("Coverage", "Mean distance to the nearest retained window, the quantity the coverage proposition names"),
    "qc_median":    ("Coverage", "Median distance to the nearest retained window"),
    "qc_max":       ("Coverage", "Largest such distance, what K-center minimises"),
    "fl_objective": ("Coverage", "Facility location, that distance through a kernel"),
    "redundancy":   ("Diversity", "Summed similarity inside the retained set"),
    "sinkhorn":     ("Distribution matching", "Entropic transport cost at uniform weights"),
    "h_tod":        ("Temporal diversity", "Hour-of-day entropy of the retained set"),
    "h_dow":        ("Temporal diversity", "Day-of-week entropy of the retained set"),
}
ORDER = ["Coverage", "Diversity", "Distribution matching", "Temporal diversity"]


def scores() -> pd.DataFrame:
    """One row per retained set, every family in the raw space."""
    d = pd.read_csv(OUT / "proxy_families_raw.csv")
    parsed = d.index_file.str.extract(
        r"^(?P<method>.+)_euclidean_(?P<ratio>\d+)_seed(?P<seed>\d+)\.json$")
    d["method"] = parsed.method
    d["ratio"] = parsed.ratio.astype(int) / 100.0
    d["coreset_seed"] = parsed.seed.astype(int)
    return d.drop(columns=["index_file"])


def accuracy() -> pd.DataFrame:
    """Realised test MAE per architecture, network, objective, budget and seed."""
    frames = [pd.read_csv(RESULT / f"{name}.csv")
              for name in ("phase_c_method_comparison", "phase_c_extra_ratios")
              if (RESULT / f"{name}.csv").exists()]
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"model": "architecture",
                          "coreset_selection_strategy": "method",
                          "coreset_selection_ratio": "ratio",
                          "MAE_mean": "mae"})
    return d[["architecture", "dataset", "method", "ratio", "coreset_seed", "mae"]]


def main() -> None:
    args = parse_args()
    merged = accuracy().merge(scores(), on=["dataset", "method", "ratio", "coreset_seed"],
                              how="inner")
    merged["ratio"] = merged.ratio.round(2)
    merged = merged[merged.ratio < 1.0]
    print(f"{len(merged)} runs carry a score in the raw space")

    present = [s for s in FAMILY if s in merged.columns]
    cell = ["architecture", "dataset", "ratio"]
    averaged = merged.groupby(cell + ["method"], as_index=False)[present + ["mae"]].mean()

    budgets = sorted(averaged.ratio.unique())
    rows = []
    for score in present:
        row = {"family": FAMILY[score][0], "score": score}
        for budget in budgets:
            per_cell = []
            for keys, group in averaged.groupby(cell):
                if keys[2] != budget:
                    continue
                group = group.dropna(subset=[score, "mae"])
                if len(group) < 3 or group[score].nunique() < 2:
                    continue
                rho = spearmanr(group[score], group.mae).statistic
                if np.isfinite(rho):
                    per_cell.append({"architecture": keys[0], "rho": rho})
            if not per_cell:
                row[budget] = np.nan
                continue
            per_cell = pd.DataFrame(per_cell)
            row[budget] = float(per_cell.groupby("architecture").rho.mean().abs().mean())
            row[f"cells_{budget}"] = len(per_cell)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no cell had enough objectives to correlate")

    body = []
    for family in ORDER:
        members = table[table.family == family]
        if members.empty:
            continue
        body.append(f"\\multicolumn{{{len(budgets) + 1}}}{{l}}{{\\textit{{{family}}}}} \\\\")
        for _, row in members.iterrows():
            cells = " & ".join("--" if np.isnan(row[b]) else f"${row[b]:.2f}$" for b in budgets)
            body.append(f"\\quad {FAMILY[row.score][1]} & {cells} \\\\")
        body.append("\\addlinespace")
    if body and body[-1] == "\\addlinespace":
        body.pop()

    header = " & ".join(f"$r={b:g}$" for b in budgets)
    lines = [
        "% Generated by scripts/analysis/build_proxy_family_table.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\caption{Pre-training scores against realised test MAE, grouped by the property "
        "they measure and reported per budget. All are computed in the one raw space the "
        "coverage proposition of the main text attaches to, so a gap between families is "
        "not a gap between "
        "representations. Each entry is the rank correlation with test MAE over the six "
        "seed-averaged objectives in a cell, taken over the ten architecture--network cells "
        "at that budget, averaged within architecture and reported as the absolute value "
        "over the five. The four coverage scores are one distance under different summaries; "
        "the median is zero for every objective at the two largest budgets. No family "
        "separates where reduction is aggressive, and the ordering at milder budgets is not "
        "consistent across them.}",
        "\\label{tab:proxy_families}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(budgets) + "}",
        "\\toprule",
        f"Score & {header} \\\\",
        "\\midrule",
    ] + body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    if not args.dry_run:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        TABLE.write_text("\n".join(lines) + "\n")
    table.to_csv(OUT / "proxy_families.csv", index=False)
    print(table[["family", "score"] + budgets].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nfamily mean by budget:")
    print(table.groupby("family")[budgets].mean().round(3).to_string())
    print(f"\n{'would write' if args.dry_run else 'wrote'} {TABLE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
