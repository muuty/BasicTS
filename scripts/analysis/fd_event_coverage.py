#!/usr/bin/env python3
"""Does a larger breakdown budget buy new events, or more windows of the same ones?

Training on more breakdown pairs stops helping somewhere below the largest share
a window budget can hold.  One reading of that plateau is that the extra windows
come from breakdown events the subset already covers, so they add little the
model has not already seen.

Breakdown pairs group into events that are connected in time, following the
definition in ``fd_spatial_synchronisation.py``.  For each selected set this
script reports the share of breakdown pairs it holds, the share of events it
touches, and how many windows it holds per touched event.  A selection that
raises the pair share while leaving the event share flat is buying duplicates.

The ranking that builds the subsets scores a window by its breakdown count,
which is a modular set function and counts a second window of the same event as
fully as the first.  Event coverage is submodular in the selected set, so the
two objectives separate exactly where the duplicates begin.  This script
measures that separation and needs no trained model.

Usage:  python3 fd_event_coverage.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fd_sensor_resolved import (
    DATASETS,
    N_WINDOWS,
    OUT,
    REPO,
    classify_observations,
    load_raw,
    transition_codes,
)

BREAKDOWN = 1
GAPS = (1, 3)
SELECTIONS = [
    ("Random", "random_euclidean_010_seed42.json"),
    ("Stride", "stride_euclidean_010_seed42.json"),
    ("K-medoids", "k_medoids_euclidean_010_seed42.json"),
    ("Stride + breakdown (8:2)", "fd_hyb0802_euclidean_010_seed42.json"),
    ("Stride + breakdown (6:4)", "fd_hyb0604_euclidean_010_seed42.json"),
    ("Maximum (tau=0.10)", "fd_front10_euclidean_010_seed42.json"),
]


def events_of(active: np.ndarray, gap: int) -> list[np.ndarray]:
    """Windows holding the state, grouped into runs separated by more than gap."""
    if active.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(active) > gap)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [active.size]])
    return [active[b:e] for b, e in zip(starts, stops)]


def load_indices(dataset: str, name: str) -> np.ndarray | None:
    path = REPO / "coreset_indices" / dataset / name
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("indices", next(iter(payload.values())))
    return np.asarray(payload, dtype=int)


def main() -> None:
    fits = pd.read_csv(OUT / "fd_sensor_resolved_fits.csv")
    rows = []
    for dataset in DATASETS:
        raw = load_raw(dataset)
        accepted = fits[(fits.dataset == dataset) & fits.identifiable] \
            .sort_values("sensor").reset_index(drop=True)
        code = transition_codes(classify_observations(raw, accepted), 2 / 3, 0.75)
        counts = (code == BREAKDOWN).sum(axis=1)          # breakdown detectors per window
        total_pairs = int(counts.sum())
        active = np.flatnonzero(counts > 0)

        print(f"\n===== {dataset} =====")
        print(f"windows {N_WINDOWS}, breakdown pairs {total_pairs:,}, "
              f"windows holding breakdown {active.size:,}")

        for gap in GAPS:
            groups = events_of(active, gap)
            owner = np.full(N_WINDOWS, -1, int)
            for e, windows in enumerate(groups):
                owner[windows] = e
            print(f"\n  gap {gap}: {len(groups)} events, "
                  f"median {np.median([len(g) for g in groups]):.0f} windows each")

            # the ranking the subsets are built from, as a reference curve
            order = np.argsort(-counts, kind="stable")
            cum_pairs = np.cumsum(counts[order]) / total_pairs
            seen, cum_events = set(), []
            for i in order:
                if counts[i] > 0:
                    seen.add(owner[i])
                cum_events.append(len(seen) / len(groups))
            cum_events = np.asarray(cum_events)
            print("    breakdown-ranked prefix   pair share -> event share")
            for target in (0.10, 0.20, 0.30, 0.407, 0.50, 0.607):
                j = int(np.searchsorted(cum_pairs, target))
                if j < len(cum_pairs):
                    print(f"      pairs {target:.3f} at {j + 1:5d} windows "
                            f"({(j + 1) / N_WINDOWS:.3f} of budget)   events {cum_events[j]:.3f}")

            print("    selected sets")
            for label, fname in SELECTIONS:
                idx = load_indices(dataset, fname)
                if idx is None:
                    print(f"      {label:26s} [missing]")
                    continue
                pair_share = counts[idx].sum() / total_pairs
                touched = {owner[i] for i in idx if counts[i] > 0}
                event_share = len(touched) / len(groups)
                held = sum(1 for i in idx if counts[i] > 0)
                per_event = held / max(len(touched), 1)
                print(f"      {label:26s} pairs {pair_share:.3f}   events {event_share:.3f}   "
                      f"windows per touched event {per_event:.2f}")
                rows.append({
                    "dataset": dataset, "gap_tolerance_windows": gap, "selection": label,
                    "breakdown_pair_share": float(pair_share),
                    "event_share": float(event_share),
                    "windows_per_touched_event": float(per_event),
                    "n_events": len(groups), "n_breakdown_pairs": total_pairs,
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fd_event_coverage.csv", index=False)
    print(f"\nwrote {OUT / 'fd_event_coverage.csv'}")


if __name__ == "__main__":
    main()
