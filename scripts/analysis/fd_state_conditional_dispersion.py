#!/usr/bin/env python3
"""Conditional dispersion of the future, in the metric the coverage audit uses.

The coverage audit (fd_sensor_resolved.py) reports, per detector and traffic
state, the mean history distance J_X to the assigned coreset window and the mean
future gap g between their 12-step trajectories.  Both are mean absolute
differences per element on per-detector standardised flow--occupancy--speed
windows.

The energy-score audit (fd_state_energy_score.py) measures the dispersion of the
conditional future, but on RAW flow under ||.||_2 / sqrt(H).  The two are
therefore not comparable, so the share of g that no selection can remove has
never been quantified.

This script closes that gap.  For each accepted detector and each state it
estimates

    D_p(x_i) = E_{y, y' ~ p(.|x_i)} ||y - y'||_{1,av}

by the mean pairwise distance among the futures of the k nearest training
histories of x_i, in exactly the metric and standardisation the coverage audit
uses.  Two independent draws from a distribution are at least as far apart as
half the sum of the two dispersions (energy distance is non-negative for a
metric of negative type, and mean-absolute-difference on R^d is one), so

    g  >=  (1/2) E[D_p]

holds for every coreset, budget and objective.  The reported floor share
(1/2)E[D_p] / g is therefore the part of the future gap that selection over
histories cannot reduce, and 1 - share bounds what a better objective can win.

Queries are stratified by state and subsampled per detector; candidates are
always the full training window set, so the neighbour radius is unbiased.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fd_sensor_resolved as fr  # noqa: E402

OUT = fr.OUT
STATE_NAMES = fr.STATE_NAMES


def accepted_detectors(raw: np.ndarray) -> pd.DataFrame:
    screens, _ = fr.fit_all_detectors(
        raw,
        n_bins=40,
        min_branch_bins=6,
        min_free_rise=0.10,
        min_congested_drop=0.05,
        min_fit_improvement=0.10,
    )
    return screens[screens["identifiable"]].reset_index(drop=True)


def dispersion_for_detector(
    hist: torch.Tensor,
    fut: torch.Tensor,
    codes: np.ndarray,
    k: int,
    n_query: int,
    rng: np.random.Generator,
    chunk: int,
) -> list[dict]:
    """hist/fut are [n_windows, 36] for one detector; codes is [n_windows]."""
    rows = []
    dim = hist.shape[1]
    for code, name in STATE_NAMES.items():
        idx = np.flatnonzero(codes == code)
        if idx.size == 0:
            continue
        if idx.size > n_query:
            idx = rng.choice(idx, size=n_query, replace=False)
        q = hist[idx]
        disp = np.empty(len(idx), dtype=np.float64)
        radius = np.empty(len(idx), dtype=np.float64)
        for begin in range(0, len(idx), chunk):
            end = min(begin + chunk, len(idx))
            d = torch.cdist(q[begin:end], hist, p=1) / dim
            # exclude the query itself: its own distance is exactly zero
            near = torch.topk(d, k + 1, largest=False).indices[:, 1:]
            nf = fut[near]                                    # [b, k, 36]
            pw = (nf[:, :, None, :] - nf[:, None, :, :]).abs().mean(-1)
            # mean over the k(k-1) ordered off-diagonal pairs
            s = pw.sum((1, 2)) / (k * (k - 1))
            disp[begin:end] = s.numpy()
            radius[begin:end] = torch.gather(d, 1, near).mean(1).numpy()
        rows.append(
            {
                "traffic_state_transition": name,
                "k": k,
                "n_query": int(len(idx)),
                "n_atoms_total": int(np.count_nonzero(codes == code)),
                "D_p": float(disp.mean()),
                "D_p_sd": float(disp.std()),
                "floor": float(disp.mean()) / 2.0,
                "knn_radius": float(radius.mean()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-query", type=int, default=400, help="per detector per state")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit-detectors", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=list(fr.DATASETS))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "fd_state_conditional_dispersion.csv"))
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(42)
    all_rows = []

    for dataset in args.datasets:
        raw = fr.load_raw(dataset)
        accepted = accepted_detectors(raw)
        sensors = accepted["sensor"].to_numpy(int)
        if args.limit_detectors:
            sensors = sensors[: args.limit_detectors]
            accepted = accepted.iloc[: args.limit_detectors]
        print(f"[{dataset}] {len(sensors)} accepted detectors", flush=True)

        obs = fr.classify_observations(raw, accepted)
        codes = fr.transition_codes(obs, min_valid_fraction=2 / 3, min_agreement=0.75)
        z = fr.standardise_per_detector(raw, sensors)
        history, future = fr.local_windows(z)

        for local_s, sensor in enumerate(sensors):
            h = torch.from_numpy(np.ascontiguousarray(history[local_s]))
            f = torch.from_numpy(np.ascontiguousarray(future[local_s]))
            rows = dispersion_for_detector(
                h, f, codes[:, local_s], args.k, args.n_query, rng, args.chunk
            )
            for r in rows:
                r["dataset"] = dataset
                r["sensor"] = int(sensor)
            all_rows.extend(rows)
            if (local_s + 1) % 10 == 0:
                print(f"  {local_s + 1}/{len(sensors)}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")
    print(
        df.groupby("traffic_state_transition")[["D_p", "floor", "knn_radius"]]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
