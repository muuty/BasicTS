#!/usr/bin/env python3
"""Rebuild the incident diagnostic using the main-table phase precedence.

The raw cache may contain checkpoint reruns from ablations and older phases.
This script keeps only the phase that supplies each cell in the paper's main
MAE grid, averages duplicate checkpoint reruns within a selection seed, and
then averages the available selection seeds.  It does not rescan checkpoints.
"""

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "experiments" / "result" / "analysis" / "incident_test_runs.csv"
OUTPUT = (
    REPO / "experiments" / "result" / "analysis" / "incident_main_grid_by_cell.csv"
)

MODELS = {"STGCNChebGraphConv", "AGCRN", "DCRNN", "STID", "STAEformer"}
METHODS = {"random", "recent", "stride", "graph_cut", "k_center", "k_medoids"}
RATIOS = {0.1, 0.3, 0.5, 0.7, 0.9}


def wanted_phase(row: pd.Series) -> str:
    if row["method"] == "k_medoids":
        return "phase_d_kmedoids_rerun"
    if row["model"] == "DCRNN":
        return "phase_c_dcrnn_no_cl"
    if row["ratio"] in (0.1, 0.9):
        return "phase_c_extra_ratios"
    return "phase_c_method_comparison"


def build() -> pd.DataFrame:
    raw = pd.read_csv(INPUT)
    raw = raw[
        raw["model"].isin(MODELS)
        & raw["method"].isin(METHODS)
        & raw["ratio"].isin(RATIOS)
        & raw["dataset"].isin(["SAN_BERNARDINO", "CONTRA_COSTA"])
    ].copy()
    raw["wanted_phase"] = raw.apply(wanted_phase, axis=1)
    raw = raw[raw["phase"].eq(raw["wanted_phase"])]

    seed_key = ["model", "dataset", "method", "ratio", "seed", "phase"]
    by_seed = (
        raw.groupby(seed_key, as_index=False)
        .agg(
            overall_MAE=("overall_MAE", "mean"),
            non_incident_MAE=("non_incident_MAE", "mean"),
            incident_MAE=("all_incident_MAE", "mean"),
            n_checkpoint_reruns=("run_dir", "size"),
        )
    )
    by_seed["incident_minus_nonincident"] = (
        by_seed["incident_MAE"] - by_seed["non_incident_MAE"]
    )
    by_seed["incident_minus_nonincident_pct"] = (
        100
        * by_seed["incident_minus_nonincident"]
        / by_seed["non_incident_MAE"]
    )

    key = ["model", "dataset", "method", "ratio", "phase"]
    return (
        by_seed.groupby(key, as_index=False)
        .agg(
            overall_MAE=("overall_MAE", "mean"),
            non_incident_MAE=("non_incident_MAE", "mean"),
            incident_MAE=("incident_MAE", "mean"),
            incident_minus_nonincident=("incident_minus_nonincident", "mean"),
            incident_minus_nonincident_pct=(
                "incident_minus_nonincident_pct",
                "mean",
            ),
            n_selection_seeds=("seed", "nunique"),
            n_checkpoint_reruns=("n_checkpoint_reruns", "sum"),
        )
        .sort_values(["dataset", "ratio", "model", "method"])
    )


if __name__ == "__main__":
    result = build()
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} ({len(result)} cells)")
    print(result.groupby("dataset").size().to_string())
