#!/usr/bin/env python3
"""Coreset selection by stratified allocation over fundamental-diagram transitions.

The four detector-resolved transitions (free flow, breakdown, recovery, sustained
congestion) are strata.  Classical stratified sampling allocates a sample across
strata as

    pi_h(beta)  proportional to  N_h^beta * sigma_h,

with N_h the stratum's share of window--detector atoms and sigma_h its within-stratum
dispersion of the standardised 12-step future.  beta = 1 is Neyman allocation, which
minimises the variance of the pooled estimate; beta = 0 drops the exposure weight and
equalises precision across strata.  One parameter therefore spans the manuscript's
central tension: whether a state matters in proportion to how often it occurs.

Selection acts on timestamp windows, and a window carries every detector at once, so
no window belongs to one stratum.  The retained set is instead built greedily to make
its atom composition match pi(beta): at each step the window that most reduces the L1
distance between the retained composition and the target is added.

Only detectors that pass the two-branch screen carry a state, so the composition is
measured on that subset; unscreened detectors are carried along by whatever windows
the screened ones select.

Writes coreset_indices/<DATASET>/fd_strat<b>_euclidean_0<r>_seed42.json, which the
runner picks up through CORESET.SELECTION_STRATEGY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fd_sensor_resolved as fr  # noqa: E402

STATES = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
INDEX_BASE = fr.REPO / "coreset_indices"


def accepted(raw):
    screens, _ = fr.fit_all_detectors(
        raw, n_bins=40, min_branch_bins=6, min_free_rise=0.10,
        min_congested_drop=0.05, min_fit_improvement=0.10)
    return screens[screens["identifiable"]].reset_index(drop=True)


def stratum_dispersion(future: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """sigma_h: mean over detectors of the within-state spread of standardised futures.

    future is [n_detectors, n_windows, 36]; codes is [n_windows, n_detectors].
    Spread is the mean per-element standard deviation, so it is in the same units as
    the audit's mean absolute differences.
    """
    sigma = np.zeros(len(STATES))
    for h in range(len(STATES)):
        per_det = []
        for d in range(future.shape[0]):
            m = codes[:, d] == h
            if m.sum() < 30:
                continue
            per_det.append(future[d][m].std(axis=0).mean())
        sigma[h] = float(np.mean(per_det)) if per_det else 0.0
    return sigma


def select_by_allocation(counts: np.ndarray, target: np.ndarray,
                         exposure: np.ndarray, k: int) -> np.ndarray:
    """Retain the k windows carrying the most mass under the target allocation.

    A window cannot be assigned to one stratum, so the allocation is realised by
    importance weighting: an atom in stratum h counts pi_h / N_h, the factor by which
    the target over- or under-represents that stratum relative to the archive.  At
    beta = 1 the weights reduce to sigma_h; as beta falls they rise on the rare
    strata.  Unlike matching a target composition, this degrades gracefully when the
    target is not reachable, which it is not for breakdown: breakdown is 1.5% of atoms
    and every window carries all detectors at once.
    """
    w = target / exposure
    score = counts @ w
    return np.sort(np.argsort(-score, kind="stable")[:k]).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs="*", default=[1.0, 0.5, 0.0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for dataset in fr.DATASETS:
        raw = fr.load_raw(dataset)
        acc = accepted(raw)
        sensors = acc["sensor"].to_numpy(int)
        obs = fr.classify_observations(raw, acc)
        codes = fr.transition_codes(obs, min_valid_fraction=2 / 3, min_agreement=0.75)
        z = fr.standardise_per_detector(raw, sensors)
        _, future = fr.local_windows(z)

        counts = np.stack([(codes == h).sum(axis=1) for h in range(len(STATES))], 1).astype(float)
        exposure = counts.sum(0) / counts.sum()
        sigma = stratum_dispersion(future, codes)
        k = int(round(args.ratio * codes.shape[0]))

        print(f"\n=== {dataset}  {len(sensors)} detectors, {codes.shape[0]} windows, keep {k}")
        print("  exposure  " + "  ".join(f"{s}={e:.4f}" for s, e in zip(STATES, exposure)))
        print("  sigma     " + "  ".join(f"{s}={v:.4f}" for s, v in zip(STATES, sigma)))

        for beta in args.betas:
            w = (exposure ** beta) * sigma
            target = w / w.sum()
            idx = select_by_allocation(counts, target, exposure, k)
            got = counts[idx].sum(0) / counts[idx].sum()
            tag = f"{beta:.1f}".replace(".", "")
            name = (f"fd_strat{tag}_euclidean_"
                    f"{args.ratio:.2f}".replace(".", "") + f"_seed{args.seed}.json")
            out = INDEX_BASE / dataset / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([int(i) for i in idx]))
            print(f"  beta={beta:<4} target " + " ".join(f"{t:.3f}" for t in target)
                  + " | achieved " + " ".join(f"{a:.3f}" for a in got)
                  + f" | breakdown atoms kept {counts[idx][:,1].sum()/counts[:,1].sum():.3f}"
                  + f" -> {out.name}")


if __name__ == "__main__":
    main()
