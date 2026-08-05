#!/usr/bin/env python3
"""Group-balanced coreset selection when the groups do not partition the examples.

Class-aware pruning allocates the budget across groups and samples within them
(DRoP, Vysogorets et al., ICLR 2025; and the class-proportional coreset line).  Every
such method assumes one training example carries one group label.  A network-level
forecasting example is a time window that carries one traffic state per detector, so
the groups partition the window--detector atoms and not the windows, and no budget can
buy one group without buying the detectors attached to it.

This script realises a target allocation in that setting and records how far it gets.
With N_h the archive share of atoms in state h, the target family is

    pi_h(beta)  proportional to  N_h^beta,      atom weight  N_h^(beta-1),

    beta = 1  frequency-proportional, every atom weighs the same, which reduces to
              random selection and is the control;
    beta = 0  group-balanced, an atom weighs the inverse of its state's frequency,
              the standard worst-group weighting.

A window is scored by the mean weight of the atoms it carries, so windows dense in
rare states rise:

    score(w) = sum_h N_h^(beta-1) c_h(w) / sum_h c_h(w).

STATE LABELS FOR EVERY DETECTOR.  The two-branch flow-occupancy screen used by the
audit retains 12.5% of detectors, and it retains them for having a congested branch,
so allocation weights computed on that subset are measured on a congestion-enriched
sample.  Selection instead labels every detector by speed against its own free-flow
speed, congested when v < alpha * v_free with v_free the 85th percentile of the
detector's training speed.  That needs no two-branch fit.  alpha is fixed by agreement
with the calibrated labels on the screened detectors, so it is not a free knob.

Writes coreset_indices/<DATASET>/fd_grp<b>_euclidean_0<r>_seed42.json.
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
ALPHA_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FREE_PCTL = 85.0


def screened(raw):
    s, _ = fr.fit_all_detectors(
        raw, n_bins=40, min_branch_bins=6, min_free_rise=0.10,
        min_congested_drop=0.05, min_fit_improvement=0.10)
    return s[s["identifiable"]].reset_index(drop=True)


def speed_states(raw: np.ndarray, alpha: float, sensors: np.ndarray | None = None):
    """-1 unresolved, 0 free, 1 congested, from speed against the detector's own free-flow speed."""
    v = raw[:, :, 2] if sensors is None else raw[:, sensors, 2]
    valid = v > 0
    vf = np.array([np.percentile(v[valid[:, j], j], FREE_PCTL) if valid[:, j].any() else 0.0
                   for j in range(v.shape[1])], dtype=float)[None, :]
    state = np.full(v.shape, -1, np.int8)
    ok = valid & (vf > 0)
    state[ok & (v >= alpha * vf)] = 0
    state[ok & (v < alpha * vf)] = 1
    return state


def calibrate_alpha(raw, acc) -> tuple[float, float]:
    """Pick alpha by agreement with the calibrated labels, on screened detectors only."""
    sensors = acc["sensor"].to_numpy(int)
    fd = fr.classify_observations(raw, acc)
    resolved = fd >= 0
    best, best_a = -1.0, ALPHA_GRID[0]
    for a in ALPHA_GRID:
        sp = speed_states(raw, a, sensors)
        agree = float((sp[resolved] == fd[resolved]).mean())
        print(f"    alpha={a:.2f}  agreement with calibrated labels = {agree:.4f}")
        if agree > best:
            best, best_a = agree, a
    return best_a, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs="*", default=[1.0, 0.5, 0.0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for dataset in fr.DATASETS:
        raw = fr.load_raw(dataset)
        acc = screened(raw)
        print(f"\n=== {dataset}: {raw.shape[1]} detectors, {len(acc)} pass the two-branch screen")
        alpha, agree = calibrate_alpha(raw, acc)
        print(f"  alpha = {alpha:.2f} (agreement {agree:.4f})")

        obs = speed_states(raw, alpha)                    # every detector
        codes = fr.transition_codes(obs, min_valid_fraction=2 / 3, min_agreement=0.75)
        counts = np.stack([(codes == h).sum(1) for h in range(len(STATES))], 1).astype(float)
        tot = counts.sum(1)
        N = counts.sum(0) / counts.sum()
        k = int(round(args.ratio * codes.shape[0]))
        print("  network-wide exposure  " + "  ".join(f"{s}={e:.4f}" for s, e in zip(STATES, N)))
        print(f"  resolved atoms {counts.sum():,.0f} of {codes.size:,} "
              f"({counts.sum()/codes.size:.3f}); keep {k} of {codes.shape[0]} windows")

        for beta in args.betas:
            w = N ** (beta - 1.0)
            score = np.divide(counts @ w, tot, out=np.zeros_like(tot), where=tot > 0)
            score = score + rng.uniform(0, 1e-9, size=score.shape)      # random tiebreak
            idx = np.sort(np.argsort(-score, kind="stable")[:k]).astype(int)
            got = counts[idx].sum(0) / counts[idx].sum()
            tag = f"{beta:.1f}".replace(".", "")
            name = f"fd_grp{tag}_euclidean_" + f"{args.ratio:.2f}".replace(".", "") + \
                   f"_seed{args.seed}.json"
            out = INDEX_BASE / dataset / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([int(i) for i in idx]))
            print(f"  beta={beta:<4} composition " + " ".join(f"{a:.4f}" for a in got)
                  + f" | breakdown atoms kept {counts[idx][:, 1].sum()/counts[:, 1].sum():.3f}"
                  + f" | enrichment {got[1]/N[1]:.2f}x -> {out.name}")


if __name__ == "__main__":
    main()
