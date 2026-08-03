#!/usr/bin/env python3
"""Split-budget selection with a separate head for each transition state.

The single-head split raises breakdown exposure and lowers recovery exposure, from
0.098 under uniform sampling to 0.064, so the manuscript's expectation that closer
matching helps at recovery is never tested.  This selector gives each transition
state its own budget line,

    r = r_uniform + r_breakdown + r_recovery,

spending the uniform part on evenly spaced windows and each head on the windows
carrying the most simultaneous detectors in that state.  Both heads use the state
labels of the chronological training split, so the whole construction is computable
before any model is fitted.

The heads are affordable because each transition state is concentrated over windows:
the top decile of windows carries 60.7% of breakdown atoms and 71.8% of recovery
atoms on San Bernardino, against 10% for an even spread.  Free flow (17.7%) and
sustained congestion (36.1%) are too diffuse to buy this way.

Writes coreset_indices/<DATASET>/fd_hyb<uu><bb><rr>_euclidean_0<r>_seed42.json.
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
from fd_two_state_split import STATES, select_two  # noqa: E402


def main() -> None:
    args = parse_args()
    screen = types.SimpleNamespace(
        n_bins=40, min_branch_bins=6, min_free_rise=0.10, min_congested_drop=0.05,
        min_fit_improvement=0.10, min_valid_fraction=2 / 3, min_agreement=0.75)

    for dataset in args.datasets:
        _, transition, _ = sync.label_matrix(dataset, screen)
        n_windows = transition.shape[0]
        counts = {c: (transition == c).sum(axis=1).astype(float) for c in STATES}
        totals = {c: counts[c].sum() for c in STATES}

        idx = select_two(counts, n_windows, args.ratio, args.breakdown, args.recovery)
        uniform = round(args.ratio - args.breakdown - args.recovery, 4)
        tag = (f"fd_hyb{int(round(100 * uniform)):02d}"
               f"{int(round(100 * args.breakdown)):02d}"
               f"{int(round(100 * args.recovery)):02d}")
        ratio_str = f"{args.ratio:.2f}".replace(".", "")
        out_dir = REPO / "coreset_indices" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{tag}_euclidean_{ratio_str}_seed{args.seed}.json"
        path.write_text(json.dumps([int(i) for i in idx]))

        pi = {name: counts[c][idx].sum() / totals[c] for c, name in STATES.items()}
        print(f"[{dataset}] {tag}  n={idx.size}  "
              + "  ".join(f"{k}={v / args.ratio:.2f}x" for k, v in pi.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--breakdown", type=float, default=0.03)
    parser.add_argument("--recovery", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
