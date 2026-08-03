#!/usr/bin/env python3
"""Split-budget selection: uniform temporal coverage plus a breakdown-ranked head.

The stratified and group-balanced selectors buy breakdown exposure by spending the
whole budget on congestion-dense windows, which removes off-peak and weekend windows
and shifts the training measure toward the congested branch.  Both effects are
confounded in a single knob.

This selector separates them.  The budget r is split into

    r = r_uniform + r_head,

where the first part is the Stride objective already in the manuscript (evenly spaced
windows, uniform in calendar time) and the second ranks the remaining windows by the
number of simultaneous breakdown atoms they carry, the same count used by the spatial
concentration analysis.  The uniform part fixes calendar coverage; the head buys
breakdown exposure at the smallest congestion mass that the window unit allows.

State labels come from the two-branch flow--occupancy screen fitted on the training
split (fd_sensor_resolved), so the ranking uses no test-period information and is
computable before any model is trained.

Writes coreset_indices/<DATASET>/fd_hyb<uu><hh>_euclidean_0<r>_seed42.json, where uu
and hh are the two sub-budgets in percent, which the runner resolves through
CORESET.SELECTION_STRATEGY.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import fd_spatial_synchronisation as sync  # noqa: E402
from fd_sensor_resolved import DATASETS  # noqa: E402

BREAKDOWN_CODE = 1


def stride_indices(n_windows: int, k: int) -> np.ndarray:
    """Evenly spaced windows, identical to coreset.stride.StrideSelection."""
    return np.linspace(0, n_windows - 1, k, dtype=int)


def breakdown_counts(dataset: str, args) -> tuple[np.ndarray, int]:
    """Simultaneous breakdown detectors per training window."""
    _, transition, _ = sync.label_matrix(dataset, args)
    counts = (transition == BREAKDOWN_CODE).sum(axis=1).astype(float)
    return counts, transition.shape[0]


def select(counts: np.ndarray, n_windows: int, ratio: float, uniform: float) -> np.ndarray:
    """Union of the uniform part and the breakdown-ranked head, at the exact budget."""
    k_total = int(round(ratio * n_windows))
    k_uniform = int(round(uniform * n_windows))
    if k_uniform > k_total:
        raise ValueError("uniform sub-budget exceeds the total budget")

    kept = stride_indices(n_windows, k_uniform)
    remaining = np.setdiff1d(np.arange(n_windows), kept, assume_unique=False)
    k_head = k_total - kept.size
    # Stable order: highest breakdown count first, earliest window breaking ties.
    head_order = remaining[np.argsort(-counts[remaining], kind="stable")]
    head = head_order[:k_head]
    return np.sort(np.concatenate([kept, head])).astype(int)


def report(tag: str, idx: np.ndarray, counts: np.ndarray, n_windows: int) -> None:
    total = counts.sum()
    k = int(round(0.10 * n_windows))
    oracle = np.argsort(-counts, kind="stable")[:k]
    print(f"  {tag:10s} n={idx.size:5d}  breakdown atoms retained="
          f"{counts[idx].sum() / total:.3f}  (oracle at r=0.1 {counts[oracle].sum() / total:.3f})",
          flush=True)


def main() -> None:
    args = parse_args()
    screen = types.SimpleNamespace(
        n_bins=args.n_bins, min_branch_bins=args.min_branch_bins,
        min_free_rise=args.min_free_rise, min_congested_drop=args.min_congested_drop,
        min_fit_improvement=args.min_fit_improvement,
        min_valid_fraction=args.min_valid_fraction, min_agreement=args.min_agreement,
    )
    for dataset in args.datasets:
        counts, n_windows = breakdown_counts(dataset, screen)
        print(f"[{dataset}] windows={n_windows} breakdown atoms={int(counts.sum())} "
              f"windows with any={int((counts > 0).sum())}", flush=True)
        out_dir = REPO / "coreset_indices" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for uniform in args.uniform:
            idx = select(counts, n_windows, args.ratio, uniform)
            tag = (f"fd_hyb{int(round(100 * uniform)):02d}"
                   f"{int(round(100 * (args.ratio - uniform))):02d}")
            ratio_str = f"{args.ratio:.2f}".replace(".", "")
            path = out_dir / f"{tag}_euclidean_{ratio_str}_seed{args.seed}.json"
            path.write_text(json.dumps([int(i) for i in idx]))
            report(tag, idx, counts, n_windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--uniform", nargs="+", type=float, default=(0.09, 0.08, 0.06),
                        help="Sub-budget spent on uniform temporal coverage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--min-branch-bins", type=int, default=6)
    parser.add_argument("--min-free-rise", type=float, default=0.10)
    parser.add_argument("--min-congested-drop", type=float, default=0.05)
    parser.add_argument("--min-fit-improvement", type=float, default=0.10)
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    main()
