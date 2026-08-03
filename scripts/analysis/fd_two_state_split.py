#!/usr/bin/env python3
"""Can a split budget raise breakdown and recovery together?

Section 6.5 reports an asymmetry between the two transition states: at recovery the
future divergence is proportionate to how far the assigned history sits, at breakdown
it is not, so closer matching is expected to help at recovery and not at breakdown.
The split-budget objective as run lowers recovery coverage from 0.098 to 0.064, so
that expectation is untested.

This script asks, before any training, whether a three-way split

    r = r_uniform + r_breakdown + r_recovery

can raise both transition states while leaving free flow and sustained congestion at
or below the neutral level r, which is what uniform window sampling retains. It also
reports how concentrated each state is over windows, since a state that is spread
evenly cannot be bought from a small slice.

Writes experiments/result/analysis/fd_two_state_split.csv.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import fd_spatial_synchronisation as sync  # noqa: E402
from select_fd_hybrid import stride_indices  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"
STATES = {0: "free", 1: "breakdown", 2: "recovery", 3: "sustained"}


def select_two(counts: dict, n_windows: int, ratio: float,
               r_bd: float, r_rec: float) -> np.ndarray:
    """Uniform part, then a breakdown-ranked head, then a recovery-ranked head."""
    k_total = int(round(ratio * n_windows))
    k_bd = int(round(r_bd * n_windows))
    k_rec = int(round(r_rec * n_windows))
    k_uniform = k_total - k_bd - k_rec
    if k_uniform < 0:                      # rounding can overshoot by one window
        k_rec += k_uniform
        k_uniform = 0

    kept = stride_indices(n_windows, k_uniform) if k_uniform else np.array([], int)
    for code, k_head in ((1, k_bd), (2, k_rec)):
        if k_head <= 0:
            continue
        remaining = np.setdiff1d(np.arange(n_windows), kept)
        order = remaining[np.argsort(-counts[code][remaining], kind="stable")]
        kept = np.concatenate([kept, order[:k_head]])
    return np.sort(kept).astype(int)


def concentration(counts: np.ndarray, frac: float = 0.10) -> float:
    k = int(round(frac * counts.size))
    return float(np.sort(counts)[::-1][:k].sum() / counts.sum())


def main() -> None:
    args = parse_args()
    screen = types.SimpleNamespace(
        n_bins=40, min_branch_bins=6, min_free_rise=0.10, min_congested_drop=0.05,
        min_fit_improvement=0.10, min_valid_fraction=2 / 3, min_agreement=0.75)

    rows = []
    for dataset in args.datasets:
        _, transition, _ = sync.label_matrix(dataset, screen)
        n_windows = transition.shape[0]
        counts = {c: (transition == c).sum(axis=1).astype(float) for c in STATES}
        totals = {c: counts[c].sum() for c in STATES}

        print(f"\n=== {dataset}: share of a state's atoms in its own top 10% of windows ===")
        for c, name in STATES.items():
            print(f"  {name:11s}{concentration(counts[c]):6.3f}   "
                  f"(even-over-windows would be 0.100)")

        r = args.ratio
        grid = [(round(b, 3), round(v, 3))
                for b in np.arange(0.0, r + 1e-9, r / 10)
                for v in np.arange(0.0, r + 1e-9, r / 10)
                if b + v <= r + 1e-9]
        print(f"\n=== {dataset}: three-way split at r={r} "
              f"(pi_h / r; admissible = free and sustained at or below 1) ===")
        print(f"{'r_bd':>6}{'r_rec':>7}{'free':>8}{'brkdn':>8}{'recov':>8}{'sust':>8}   ok")
        for r_bd, r_rec in grid:
            idx = select_two(counts, n_windows, r, r_bd, r_rec)
            pi = {name: float(counts[c][idx].sum() / totals[c]) for c, name in STATES.items()}
            rho = {k: v / r for k, v in pi.items()}
            ok = (rho["free"] <= 1.0 and rho["sustained"] <= 1.0
                  and rho["breakdown"] > 1.0 and rho["recovery"] > 1.0)
            rows.append({"dataset": dataset, "ratio": r, "r_breakdown": r_bd,
                         "r_recovery": r_rec, **{f"rho_{k}": v for k, v in rho.items()},
                         "admissible": ok})
            if r_bd + r_rec > 0:
                print(f"{r_bd:6.2f}{r_rec:7.2f}{rho['free']:8.2f}{rho['breakdown']:8.2f}"
                      f"{rho['recovery']:8.2f}{rho['sustained']:8.2f}   {'yes' if ok else ''}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fd_two_state_split.csv", index=False)
    print(f"\nwrote {OUT / 'fd_two_state_split.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["SAN_BERNARDINO"])
    parser.add_argument("--ratio", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
