#!/usr/bin/env python3
"""Held-out validation of quantization cost using archived Phase C results.

The analysis requires no forecasting-model retraining.  It averages the three
selection seeds for each (model, dataset, ratio, method) cell, then compares
all method pairs within a fixed (model, dataset, ratio) block.  A comparison is
concordant when the method with lower quantization cost also has lower test
MAE.  Dataset and model folds hold out complete blocks.  Method folds evaluate
only pairs containing the held-out method; pairs among the remaining methods
form the development partition.

Outputs:
  experiments/result/analysis/proxy_heldout_folds.csv
  experiments/result/analysis/proxy_heldout_summary.csv
  experiments/result/analysis/proxy_heldout_overall.csv
  experiments/result/analysis/proxy_raw_l1_correlations.csv
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments" / "result"
OUTPUT_DIR = RESULT_DIR / "analysis"
PROXY_DIR = ROOT / "coreset_indices"

METHODS = {
    "graph_cut",
    "k_center",
    "k_medoids",
    "random",
    "recent",
    "stride",
}
DATASETS = ("CONTRA_COSTA", "SAN_BERNARDINO")
KEYS = ["model", "dataset", "ratio", "method"]
BLOCK_KEYS = ["model", "dataset", "ratio"]


def load_results() -> pd.DataFrame:
    """Reconstruct the Phase C grid used by the paper."""
    base = pd.read_csv(RESULT_DIR / "phase_c_method_comparison.csv")
    extra = pd.read_csv(RESULT_DIR / "phase_c_extra_ratios.csv")
    results = pd.concat([base, extra], ignore_index=True)

    dcrnn_path = RESULT_DIR / "phase_c_dcrnn_no_cl.csv"
    if dcrnn_path.exists():
        results = results[results["model"] != "DCRNN"]
        results = pd.concat([results, pd.read_csv(dcrnn_path)], ignore_index=True)

    phase_d_path = RESULT_DIR / "phase_d_kmedoids_rerun.csv"
    if phase_d_path.exists():
        results = results[
            results["coreset_selection_strategy"] != "k_medoids"
        ]
        results = pd.concat([results, pd.read_csv(phase_d_path)], ignore_index=True)

    results = results.rename(
        columns={
            "coreset_selection_strategy": "method",
            "coreset_selection_ratio": "ratio",
            "coreset_seed": "seed",
        }
    )
    results["ratio"] = results["ratio"].round(2)
    return results[
        (results["ratio"] < 1.0)
        & results["method"].isin(METHODS)
        & results["dataset"].isin(DATASETS)
    ].copy()


def parse_proxy_name(name: str) -> dict[str, object] | None:
    """Parse only canonical METHOD_euclidean_RATIO_seed index names."""
    stem = Path(name).stem
    match = re.fullmatch(
        r"(graph_cut|k_center|k_medoids|random|recent|stride)"
        r"_euclidean_(\d{3})_seed(\d+)",
        stem,
    )
    if match is None:
        return None
    return {
        "method": match.group(1),
        "ratio": round(int(match.group(2)) / 100.0, 2),
        "seed": int(match.group(3)),
    }


def load_proxies() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        with (PROXY_DIR / dataset / "proxy_metrics.json").open() as handle:
            raw = json.load(handle)
        for name, metrics in raw.items():
            parsed = parse_proxy_name(name)
            if parsed is None or "qc_raw_l1" not in metrics:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    **parsed,
                    "quantization_cost": metrics["qc_raw_l1"],
                    "quantization_median": metrics.get("qc_raw_l1_median"),
                }
            )
    proxies = pd.DataFrame(rows)
    duplicate = proxies.duplicated(["dataset", "method", "ratio", "seed"])
    if duplicate.any():
        raise ValueError("Canonical proxy keys are not unique")
    return proxies


def build_cells() -> pd.DataFrame:
    merged = load_results().merge(
        load_proxies(),
        on=["dataset", "method", "ratio", "seed"],
        how="inner",
        validate="many_to_one",
    )
    cells = (
        merged.groupby(KEYS, as_index=False)
        .agg(
            MAE_mean=("MAE_mean", "mean"),
            quantization_cost=("quantization_cost", "mean"),
            n_selection_seeds=("seed", "nunique"),
        )
    )
    block_sizes = cells.groupby(BLOCK_KEYS)["method"].nunique()
    complete = block_sizes[block_sizes == len(METHODS)].index
    cells = cells.set_index(BLOCK_KEYS).loc[complete].reset_index()
    if cells.groupby(BLOCK_KEYS)["method"].nunique().ne(len(METHODS)).any():
        raise ValueError("Incomplete evaluation block after filtering")
    return cells


def build_raw_correlations() -> pd.DataFrame:
    """Per-model pooled correlations for the two raw L1 summaries."""
    merged = load_results().merge(
        load_proxies(),
        on=["dataset", "method", "ratio", "seed"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for model, group in merged.groupby("model", sort=True):
        for metric in ("quantization_cost", "quantization_median"):
            valid = group[[metric, "MAE_mean"]].dropna()
            rows.append(
                {
                    "model": model,
                    "proxy_metric": metric,
                    "pearson_r": valid[metric].corr(valid["MAE_mean"], method="pearson"),
                    "spearman_r": valid[metric].corr(valid["MAE_mean"], method="spearman"),
                    "n": len(valid),
                }
            )
    return pd.DataFrame(rows)


def build_pairs(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block, group in cells.groupby(BLOCK_KEYS, sort=True):
        records = group.sort_values("method").to_dict("records")
        for left, right in itertools.combinations(records, 2):
            proxy_delta = left["quantization_cost"] - right["quantization_cost"]
            mae_delta = left["MAE_mean"] - right["MAE_mean"]
            if np.isclose(proxy_delta, 0.0) or np.isclose(mae_delta, 0.0):
                continue
            rows.append(
                {
                    **dict(zip(BLOCK_KEYS, block)),
                    "method_left": left["method"],
                    "method_right": right["method"],
                    "concordant": float(np.sign(proxy_delta) == np.sign(mae_delta)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_folds(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_values = {
        "dataset": sorted(pairs["dataset"].unique()),
        "model": sorted(pairs["model"].unique()),
        "method": sorted(METHODS),
    }
    for axis, values in fold_values.items():
        for held_out in values:
            if axis == "method":
                in_left = pairs["method_left"] == held_out
                in_right = pairs["method_right"] == held_out
                test_mask = in_left | in_right
                train_mask = ~(in_left | in_right)
            else:
                test_mask = pairs[axis] == held_out
                train_mask = ~test_mask
            train = pairs[train_mask]
            test = pairs[test_mask]
            rows.append(
                {
                    "holdout_axis": axis,
                    "held_out": held_out,
                    "development_concordance": train["concordant"].mean(),
                    "heldout_concordance": test["concordant"].mean(),
                    "generalization_gap": (
                        test["concordant"].mean() - train["concordant"].mean()
                    ),
                    "n_development_pairs": len(train),
                    "n_heldout_pairs": len(test),
                    "n_heldout_blocks": test.groupby(BLOCK_KEYS).ngroups,
                }
            )
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby("holdout_axis", as_index=False)
        .agg(
            n_folds=("held_out", "size"),
            development_concordance=("development_concordance", "mean"),
            heldout_concordance=("heldout_concordance", "mean"),
            heldout_min=("heldout_concordance", "min"),
            heldout_max=("heldout_concordance", "max"),
            mean_generalization_gap=("generalization_gap", "mean"),
            heldout_pairs_per_fold=("n_heldout_pairs", "median"),
            heldout_blocks_per_fold=("n_heldout_blocks", "median"),
        )
        .sort_values("holdout_axis")
    )


def summarise_overall(pairs: pd.DataFrame) -> pd.DataFrame:
    """Cluster bootstrap over model-dataset-ratio blocks."""
    block_scores = pairs.groupby(BLOCK_KEYS)["concordant"].mean().to_numpy()
    rng = np.random.default_rng(42)
    draws = rng.choice(
        block_scores,
        size=(20_000, len(block_scores)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return pd.DataFrame(
        [
            {
                "n_blocks": len(block_scores),
                "n_pairs": len(pairs),
                "concordance": pairs["concordant"].mean(),
                "block_bootstrap_ci_low": low,
                "block_bootstrap_ci_high": high,
                "blocks_above_half": int((block_scores > 0.5).sum()),
            }
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    pairs = build_pairs(cells)
    folds = evaluate_folds(pairs)
    summary = summarise(folds)
    overall = summarise_overall(pairs)
    raw_correlations = build_raw_correlations()

    folds.to_csv(OUTPUT_DIR / "proxy_heldout_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "proxy_heldout_summary.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "proxy_heldout_overall.csv", index=False)
    raw_correlations.to_csv(
        OUTPUT_DIR / "proxy_raw_l1_correlations.csv", index=False
    )

    print(
        f"Complete cells: {len(cells)}; blocks: "
        f"{cells.groupby(BLOCK_KEYS).ngroups}; pair decisions: {len(pairs)}"
    )
    print(folds.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
