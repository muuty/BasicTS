#!/usr/bin/env python3
"""One-factor sensitivity checks for the detector-resolved FD screen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fd_sensor_resolved import (
    DATASETS,
    N_WINDOWS,
    classify_observations,
    fit_all_detectors,
    load_raw,
    transition_codes,
)


OUT = Path(__file__).resolve().parents[2] / "experiments" / "result" / "analysis"

SPECS = [
    dict(name="main", n_bins=40, min_branch_bins=6,
         min_free_rise=0.10, min_congested_drop=0.05,
         min_fit_improvement=0.10),
    dict(name="bins30", n_bins=30, min_branch_bins=5,
         min_free_rise=0.10, min_congested_drop=0.05,
         min_fit_improvement=0.10),
    dict(name="bins50", n_bins=50, min_branch_bins=8,
         min_free_rise=0.10, min_congested_drop=0.05,
         min_fit_improvement=0.10),
    dict(name="drop03", n_bins=40, min_branch_bins=6,
         min_free_rise=0.10, min_congested_drop=0.03,
         min_fit_improvement=0.10),
    dict(name="drop10", n_bins=40, min_branch_bins=6,
         min_free_rise=0.10, min_congested_drop=0.10,
         min_fit_improvement=0.10),
    dict(name="gain05", n_bins=40, min_branch_bins=6,
         min_free_rise=0.10, min_congested_drop=0.05,
         min_fit_improvement=0.05),
    dict(name="gain20", n_bins=40, min_branch_bins=6,
         min_free_rise=0.10, min_congested_drop=0.05,
         min_fit_improvement=0.20),
]


def fit_sensitivity(dataset: str, raw: np.ndarray) -> tuple[list[dict], pd.DataFrame]:
    fitted = {}
    for spec in SPECS:
        print(f"[{dataset}] {spec['name']}", flush=True)
        screen, _ = fit_all_detectors(raw, **{k: v for k, v in spec.items() if k != "name"})
        fitted[spec["name"]] = screen

    main = fitted["main"]
    main_ids = set(main.loc[main.identifiable, "sensor"].astype(int))
    rows = []
    for spec in SPECS:
        screen = fitted[spec["name"]]
        accepted = screen[screen.identifiable]
        ids = set(accepted.sensor.astype(int))
        overlap = sorted(main_ids & ids)
        union = main_ids | ids
        if overlap:
            left = main.set_index("sensor").loc[overlap, "critical_occupancy"]
            right = screen.set_index("sensor").loc[overlap, "critical_occupancy"]
            critical_mae = float(np.abs(left.to_numpy() - right.to_numpy()).mean())
        else:
            critical_mae = np.nan
        rows.append({
            "analysis": "fd_fit",
            "dataset": dataset,
            "specification": spec["name"],
            **{k: v for k, v in spec.items() if k != "name"},
            "n_identifiable": len(ids),
            "jaccard_vs_main": len(overlap) / len(union) if union else np.nan,
            "critical_occupancy_mae_vs_main": critical_mae,
        })
    return rows, main


def window_sensitivity(dataset: str, raw: np.ndarray, main: pd.DataFrame) -> list[dict]:
    accepted = main[main.identifiable].sort_values("sensor")
    observation_state = classify_observations(raw, accepted)
    rows = []
    for valid_fraction in [0.50, 2 / 3, 0.75]:
        for agreement in [2 / 3, 0.75, 0.80]:
            transition = transition_codes(
                observation_state,
                min_valid_fraction=valid_fraction,
                min_agreement=agreement,
            )
            resolved = transition >= 0
            rows.append({
                "analysis": "window_state",
                "dataset": dataset,
                "specification": f"valid{valid_fraction:.3f}_agree{agreement:.3f}",
                "min_valid_fraction": valid_fraction,
                "min_agreement": agreement,
                "n_detectors": transition.shape[1],
                "resolved_fraction": float(resolved.mean()),
                "n_free_to_free": int((transition == 0).sum()),
                "n_breakdown": int((transition == 1).sum()),
                "n_recovery": int((transition == 2).sum()),
                "n_congested_to_congested": int((transition == 3).sum()),
                "n_possible_atoms": int(N_WINDOWS * transition.shape[1]),
            })
    return rows


def main() -> None:
    rows = []
    for dataset in DATASETS:
        raw = load_raw(dataset)
        fit_rows, main_screen = fit_sensitivity(dataset, raw)
        rows.extend(fit_rows)
        rows.extend(window_sensitivity(dataset, raw, main_screen))
    result = pd.DataFrame(rows)
    path = OUT / "fd_sensor_resolved_sensitivity.csv"
    result.to_csv(path, index=False)
    print(f"wrote {path} ({len(result)} rows)", flush=True)


if __name__ == "__main__":
    main()
