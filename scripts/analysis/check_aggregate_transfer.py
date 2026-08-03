#!/usr/bin/env python3
"""Does ranking selection objectives by aggregate MAE order their breakdown loss?

The claim is easy to state and easy to get wrong, so this script tests it from several
directions and prints all of them, including the ones that disagree.

  1  unit of analysis   one (network, architecture, budget) cell, not a pooled average
  2  objective set      all six, without Graph Cut, without both weak objectives
  3  aggregate measure  test MAE over all detectors, and the atom-weighted mean over
                        the four states on the screened detectors, which is the same
                        population the breakdown column is measured on
  4  target             breakdown MAE, and for contrast free-flow MAE
  5  null               the same statistic on shuffled objective labels within a cell
  6  strata             results split by network and by architecture

Writes experiments/result/analysis/aggregate_transfer_checks.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1].parent
OUT = REPO / "experiments" / "result" / "analysis"

ALL_SIX = ["random", "stride", "k_medoids", "recent", "k_center", "graph_cut"]
NO_GC = ["random", "stride", "k_medoids", "recent", "k_center"]
COMPETITIVE = ["random", "stride", "k_medoids", "recent"]
# Atom shares on the screened detectors, used for the like-for-like aggregate.
SHARE = {"free": 0.727, "breakdown": 0.015, "recovery": 0.018, "sustained": 0.240}


def load() -> pd.DataFrame:
    """One row per (network, architecture, budget, objective) with both aggregates."""
    d = pd.read_csv(OUT / "fd_selection_experiment_summary.csv")
    d = d[d.ratio < 1.0]
    wide = d.pivot_table(index=["dataset", "backbone", "method", "ratio"],
                         columns="state", values="mae").reset_index()
    wide["screened_aggregate"] = sum(wide[s] * w for s, w in SHARE.items())

    u = pd.read_csv(REPO / "experiments" / "result" / "unified_mae_table.csv")
    u = u[u.ratio < 1.0].rename(columns={"model": "backbone", "MAE": "overall_aggregate"})
    return wide.merge(u[["dataset", "backbone", "method", "ratio", "overall_aggregate"]],
                      on=["dataset", "backbone", "method", "ratio"], how="inner")


def cell_rho(df: pd.DataFrame, methods: list[str], x: str, y: str,
             rng=None) -> list[float]:
    """Spearman inside each cell; with rng, shuffle the pairing to get a null draw."""
    out = []
    for _, g in df[df.method.isin(methods)].groupby(["dataset", "backbone", "ratio"]):
        if len(g) < 4:
            continue
        a, b = g[x].to_numpy(float), g[y].to_numpy(float)
        if rng is not None:
            b = rng.permutation(b)
        if len(np.unique(a)) < 3 or len(np.unique(b)) < 3:
            continue
        out.append(spearmanr(a, b).statistic)
    return out


def summarise(name: str, vals: list[float]) -> dict:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return {"check": name, "cells": len(v), "median": np.median(v),
            "mean": v.mean(), "q25": np.percentile(v, 25), "q75": np.percentile(v, 75),
            "share_positive": (v > 0).mean()}


def main() -> None:
    parse_args()
    df = load()
    rng = np.random.default_rng(0)
    rows = []

    for label, methods in (("all six", ALL_SIX), ("no Graph Cut", NO_GC),
                           ("four competitive", COMPETITIVE)):
        for agg, agg_label in (("overall_aggregate", "overall MAE"),
                               ("screened_aggregate", "screened atom-weighted")):
            rows.append(summarise(f"{label} | {agg_label} -> breakdown",
                                  cell_rho(df, methods, agg, "breakdown")))
            rows.append(summarise(f"{label} | {agg_label} -> free flow",
                                  cell_rho(df, methods, agg, "free")))
        null = []
        for _ in range(200):
            null += cell_rho(df, methods, "overall_aggregate", "breakdown", rng=rng)
        rows.append(summarise(f"{label} | NULL, labels shuffled", null))

    print("=== transfer of the aggregate ordering, one Spearman per cell ===")
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))

    print("\n=== four competitive, overall MAE -> breakdown, split by budget ===")
    for ratio, g in df[df.method.isin(COMPETITIVE)].groupby("ratio"):
        v = cell_rho(g, COMPETITIVE, "overall_aggregate", "breakdown")
        print(f"  r={ratio}: median {np.median(v):+.2f}  cells {len(v)}  "
              f"positive {np.mean(np.asarray(v) > 0):.2f}")

    print("\n=== same, split by network and by architecture ===")
    for key in ("dataset", "backbone"):
        for name, g in df[df.method.isin(COMPETITIVE)].groupby(key):
            v = cell_rho(g, COMPETITIVE, "overall_aggregate", "breakdown")
            print(f"  {name:16s} median {np.median(v):+.2f}  cells {len(v)}")

    print("\n=== level check: is breakdown actually the costlier state? ===")
    lv = df[df.method.isin(COMPETITIVE)]
    full = pd.read_csv(OUT / "fd_selection_experiment_summary.csv")
    full = full[full.method == "full"].pivot_table(
        index=["dataset", "backbone"], columns="state", values="mae")
    for state in ("free", "breakdown", "recovery", "sustained"):
        inc = lv.set_index(["dataset", "backbone"])[state] - full[state]
        print(f"  {state:10s} mean increase {inc.mean():5.2f} MAE, "
              f"above free flow in {(inc.groupby(level=[0, 1]).mean() > (lv.set_index(['dataset','backbone'])['free'] - full['free']).groupby(level=[0,1]).mean()).mean():.0%} of cells")

    tab.to_csv(OUT / "aggregate_transfer_checks.csv", index=False)
    print(f"\nwrote {OUT / 'aggregate_transfer_checks.csv'}")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    main()
