#!/usr/bin/env python3
"""Rebuild the paper's main MAE grid with cell-level provenance.

The paper table is assembled from four experiment exports.  Later phases
replace earlier cells in a deliberately narrow order:

1. Phase C supplies ratios 0.3/0.5/0.7 and the full-data baseline.
2. Phase C-extra supplies ratios 0.1/0.9.
3. Phase C DCRNN-no-CL replaces every DCRNN cell, including its full baseline.
4. Phase D replaces every reduced-data K-medoids cell.

The exported rows already contain one MAE summary per coreset/selection seed.
Consequently, the standard deviation produced here is across selection seeds
with ENV.SEED fixed at 42; it is not an independent-training-seed estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "experiments" / "result"

SOURCES = {
    "phase_c": RESULT / "phase_c_method_comparison.csv",
    "phase_c_extra": RESULT / "phase_c_extra_ratios.csv",
    "phase_c_dcrnn_no_cl": RESULT / "phase_c_dcrnn_no_cl.csv",
    "phase_d": RESULT / "phase_d_kmedoids_rerun.csv",
}

MODELS = ["AGCRN", "DCRNN", "STAEformer", "STGCNChebGraphConv", "STID"]
DATASETS = ["CONTRA_COSTA", "SAN_BERNARDINO"]
METHODS = ["graph_cut", "k_center", "k_medoids", "random", "recent", "stride"]
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_source(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.rename(
        columns={
            "coreset_selection_strategy": "method",
            "coreset_selection_ratio": "ratio",
            "coreset_seed": "selection_seed",
            "MAE_mean": "MAE",
        }
    )[
        ["model", "dataset", "method", "ratio", "selection_seed", "env_seed", "MAE"]
    ].copy()
    out["source"] = name
    return out


def choose_source(row: pd.Series) -> str:
    if row["ratio"] < 1.0 and row["method"] == "k_medoids":
        return "phase_d"
    if row["model"] == "DCRNN":
        return "phase_c_dcrnn_no_cl"
    if row["ratio"] in (0.1, 0.9):
        return "phase_c_extra"
    return "phase_c"


def expected_cells() -> pd.DataFrame:
    reduced = pd.MultiIndex.from_product(
        [MODELS, DATASETS, METHODS, RATIOS],
        names=["model", "dataset", "method", "ratio"],
    ).to_frame(index=False)
    full = pd.MultiIndex.from_product(
        [MODELS, DATASETS], names=["model", "dataset"]
    ).to_frame(index=False)
    full["method"] = "full"
    full["ratio"] = 1.0
    return pd.concat([reduced, full], ignore_index=True)


def build_grid() -> pd.DataFrame:
    raw = pd.concat(
        [load_source(name, path) for name, path in SOURCES.items()],
        ignore_index=True,
    )

    # Full-data exports use the placeholder strategy "random".  It is one
    # baseline per model/dataset, not a Random-method reduced-data result.
    raw.loc[raw["ratio"].eq(1.0), "method"] = "full"
    raw["chosen_source"] = raw.apply(choose_source, axis=1)
    selected = raw[raw["source"].eq(raw["chosen_source"])].copy()

    key = ["model", "dataset", "method", "ratio", "source"]
    grid = (
        selected.groupby(key, as_index=False, sort=True)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", lambda x: x.std(ddof=1)),
            n_selection_seeds=("selection_seed", "nunique"),
            selection_seeds=(
                "selection_seed",
                lambda x: ",".join(str(v) for v in sorted(x.unique())),
            ),
            n_training_seeds=("env_seed", "nunique"),
            training_seeds=(
                "env_seed", lambda x: ",".join(str(v) for v in sorted(x.unique()))
            ),
        )
    )
    # Match pandas' legacy CSV behaviour for a singleton if one ever appears.
    grid["MAE_std"] = grid["MAE_std"].where(grid["n_selection_seeds"] > 1, np.nan)

    expected = expected_cells()
    observed = grid[["model", "dataset", "method", "ratio"]]
    missing = expected.merge(observed, how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")]
    extras = observed.merge(expected, how="left", indicator=True)
    extras = extras[extras["_merge"].eq("left_only")]
    if not missing.empty or not extras.empty:
        raise RuntimeError(
            "Unexpected paper grid.\n"
            f"Missing:\n{missing.drop(columns='_merge').to_string(index=False)}\n"
            f"Extra:\n{extras.drop(columns='_merge').to_string(index=False)}"
        )
    if not grid["n_training_seeds"].eq(1).all() or set(grid["training_seeds"]) != {"42"}:
        raise RuntimeError("Main grid is expected to use only ENV.SEED=42")
    return grid.sort_values(["model", "dataset", "method", "ratio"]).reset_index(drop=True)


def write_outputs(grid: pd.DataFrame, output_dir: Path, write_legacy: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = output_dir / "paper_main_grid_provenance.csv"
    grid.to_csv(provenance, index=False)
    print(f"wrote {provenance} ({len(grid)} cells)")

    if not write_legacy:
        return
    legacy_grid = grid.copy()
    # Preserve the historical schema: its full-data baseline was encoded as
    # method=random even though no reduced random subset is involved.
    legacy_grid.loc[legacy_grid["ratio"].eq(1.0), "method"] = "random"
    legacy = legacy_grid.rename(columns={"n_selection_seeds": "n_samples"})[
        ["model", "dataset", "method", "ratio", "MAE_mean", "MAE_std", "n_samples"]
    ]
    legacy.to_csv(output_dir / "unified_mae_with_std.csv", index=False)
    table = legacy_grid.rename(columns={"MAE_mean": "MAE"})[
        ["model", "dataset", "method", "ratio", "source", "MAE"]
    ]
    table.to_csv(output_dir / "unified_mae_table.csv", index=False)
    print("wrote legacy unified_mae_with_std.csv and unified_mae_table.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT,
        help="Directory for paper_main_grid_provenance.csv (default: experiments/result)",
    )
    parser.add_argument(
        "--write-legacy",
        action="store_true",
        help="Also replace the two legacy unified_mae*.csv files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_outputs(build_grid(), args.output_dir, args.write_legacy)
