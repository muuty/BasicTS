#!/usr/bin/env python3
"""How each selection method samples the four traffic states.

A coreset is a set of network-wide windows; selecting a window captures every
detector-atom in it. For each method and ratio we report, per state:
  base_rate    -- share of all resolved atoms in that state
  capture_rate -- share of that state's atoms that fall in selected windows
  enrichment   -- capture_rate / ratio (1 = budget-neutral, >1 over-selected)
  composition  -- share of the coreset's resolved atoms in that state
Averaged over available selection seeds and pooled over the two networks.

Model-free: uses only FD state labels and coreset indices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from fd_sensor_resolved import (  # noqa: E402
    DATASETS, OUT, classify_observations, fit_all_detectors,
    load_indices, load_raw, transition_codes,
)

METHODS = ["k_medoids", "k_center", "graph_cut", "random", "recent", "stride"]
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [42, 123, 456]
STATES = {0: "free_to_free", 1: "breakdown", 2: "recovery", 3: "congested_to_congested"}


def state_code_matrix(dataset):
    raw = load_raw(dataset)
    screens, _ = fit_all_detectors(raw, n_bins=40, min_branch_bins=6,
                                   min_free_rise=0.10, min_congested_drop=0.05,
                                   min_fit_improvement=0.10)
    accepted = screens[screens["identifiable"]].reset_index(drop=True)
    obs = classify_observations(raw, accepted)
    code = transition_codes(obs, min_valid_fraction=2 / 3, min_agreement=0.75)
    return code  # [n_windows, n_accepted], values -1..3


def main():
    per_dataset = {d: state_code_matrix(d) for d in DATASETS}
    rows = []
    for method in METHODS:
        for ratio in RATIOS:
            # accumulate atom counts across datasets and seeds
            base = {s: 0 for s in STATES}
            cap = {s: 0 for s in STATES}
            n_seeds = 0
            for seed in SEEDS:
                got = True
                cap_seed = {s: 0 for s in STATES}
                base_seed = {s: 0 for s in STATES}
                for d in DATASETS:
                    code = per_dataset[d]
                    try:
                        sel = load_indices(d, method, ratio, seed)
                    except FileNotFoundError:
                        got = False
                        break
                    for s in STATES:
                        base_seed[s] += int((code == s).sum())
                        cap_seed[s] += int((code[sel] == s).sum())
                if not got:
                    continue
                n_seeds += 1
                for s in STATES:
                    base[s] += base_seed[s]
                    cap[s] += cap_seed[s]
            if n_seeds == 0:
                continue
            total_base = sum(base.values())
            total_cap = sum(cap.values())
            for s, name in STATES.items():
                base_rate = base[s] / total_base
                capture_rate = cap[s] / base[s] if base[s] else np.nan
                composition = cap[s] / total_cap if total_cap else np.nan
                rows.append({
                    "method": method, "ratio": ratio, "n_seeds": n_seeds,
                    "state": name,
                    "base_rate": base_rate,
                    "capture_rate": capture_rate,
                    "enrichment": capture_rate / ratio if ratio else np.nan,
                    "composition": composition,
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fd_state_selection_rate.csv", index=False)
    print("wrote fd_state_selection_rate.csv\n")

    # readable summary: enrichment (capture/ratio) at r=0.1 and r=0.3
    for ratio in (0.1, 0.3):
        print(f"=== enrichment (capture_rate / ratio) at r={ratio}  "
              f"[1.0 = budget-neutral, >1 over-selected] ===")
        piv = (out[out.ratio == ratio]
               .pivot(index="method", columns="state", values="enrichment")
               .reindex(METHODS)[list(STATES.values())])
        piv.columns = ["Free", "Breakdown", "Recovery", "Congestion"]
        print(piv.round(2).to_string(), "\n")
    print("=== base rate of each state (all atoms) ===")
    br = out.groupby("state").base_rate.first().reindex(list(STATES.values()))
    for name, v in br.items():
        print(f"  {name:24} {v:.4f}")


if __name__ == "__main__":
    main()
