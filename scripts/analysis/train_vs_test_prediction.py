#!/usr/bin/env python3
"""What the quantization error predicts: training risk, its gap, or test risk.

Proposition 1 bounds the discrepancy between the risk of a fixed forecasting model
under the full empirical training measure and its risk under the retained measure.
Nothing in it concerns the held-out test period, yet the manuscript only ever scores
the quantization error against test MAE. If the bound is doing work, the association
should be strongest for the quantity it names and weaken as the target moves away
from it.

Three targets are compared against the same pre-training score, on the same runs:

  train_full_mae   risk under the full empirical training measure
  measure_gap      |train_full_mae - train_retained_mae|, the discrepancy bounded
  test_mae         risk on the held-out test period

Association is Spearman inside a fixed architecture, network and budget, so neither
the architecture nor the budget can drive it, together with the share of within-block
pairwise orderings the score gets right, which is the statistic the manuscript
reports. eval_on_train_split.py produces the input.

Writes experiments/result/analysis/train_vs_test_prediction.csv.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
INDEX = REPO / "coreset_indices"

TARGETS = ["train_full_mae", "measure_gap", "test_mae", "train_retained_mae"]
SCORES = ["qc_raw_l1", "qc_test_l1"]


def proxy_table() -> pd.DataFrame:
    """The pre-training scores, keyed by the retained set they were computed on."""
    rows = []
    for dataset in ("SAN_BERNARDINO", "CONTRA_COSTA"):
        path = INDEX / dataset / "proxy_metrics.json"
        if not path.exists():
            continue
        for name, values in json.loads(path.read_text()).items():
            if "_euclidean_" in name:
                method, rest = name.split("_euclidean_")
                ratio_text, seed_text = rest.replace(".json", "").split("_seed")
                rows.append({
                    "dataset": dataset, "method": method,
                    "ratio": int(ratio_text) / 100.0, "coreset_seed": int(seed_text),
                    **{s: values.get(s, np.nan) for s in SCORES},
                })
    return pd.DataFrame(rows)


def concordance(score: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    """Pairwise orderings the score gets right, ties in either series excluded."""
    right = total = 0
    for i, j in combinations(range(len(score)), 2):
        if score[i] == score[j] or target[i] == target[j]:
            continue
        total += 1
        right += (score[i] < score[j]) == (target[i] < target[j])
    return right, total


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(p) for p in sorted(OUT.glob(args.glob))]
    if not frames:
        raise SystemExit(f"no input matching {args.glob} in {OUT}")
    runs = pd.concat(frames, ignore_index=True)
    runs["ratio"] = runs.ratio.round(2)
    # The runner names a network by its config path, the index files by the
    # network alone.
    runs["dataset"] = runs.dataset.str.rsplit("/", n=1).str[-1]
    keys = ["dataset", "architecture", "method", "ratio", "coreset_seed"]
    before = len(runs)
    runs = runs.sort_values("ckpt_dir").drop_duplicates(keys, keep="first")
    print(f"{before} rows, {len(runs)} after collapsing repeated runs of one configuration")

    merged = runs.merge(proxy_table(), on=["dataset", "method", "ratio", "coreset_seed"],
                        how="inner")
    print(f"{len(merged)} rows carry a pre-training score")
    merged = merged[merged.ratio < 1.0]

    # The manuscript compares seed-averaged objectives, six of them per block, so the
    # pair count it reports is 15 per block. Averaging here keeps this comparable.
    if not args.per_seed:
        merged = (merged.groupby(["dataset", "architecture", "method", "ratio"],
                                 as_index=False)[SCORES + TARGETS].mean())
        print(f"{len(merged)} seed-averaged objectives")

    records = []
    for score in SCORES:
        for target in TARGETS:
            rhos, right, total, blocks = [], 0, 0, 0
            for (arch, dataset, ratio), block in merged.groupby(
                    ["architecture", "dataset", "ratio"]):
                block = block.dropna(subset=[score, target])
                if len(block) < 3:
                    continue
                blocks += 1
                rho = spearmanr(block[score], block[target]).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
                r, t = concordance(block[score].to_numpy(), block[target].to_numpy())
                right, total = right + r, total + t
            records.append({
                "score": score, "target": target, "blocks": blocks,
                "median_spearman": np.median(rhos) if rhos else np.nan,
                "mean_spearman": np.mean(rhos) if rhos else np.nan,
                "concordant": right / total if total else np.nan,
                "pairs": total,
            })

    table = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "train_vs_test_prediction.csv", index=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Per budget, since the manuscript's claim is that ranking matters only where
    # reduction is aggressive.
    print("\nby budget:")
    per_ratio = []
    for score in SCORES:
        for target in TARGETS:
            for ratio, group in merged.groupby("ratio"):
                rhos = []
                for _, block in group.groupby(["architecture", "dataset"]):
                    block = block.dropna(subset=[score, target])
                    if len(block) < 3:
                        continue
                    rho = spearmanr(block[score], block[target]).statistic
                    if np.isfinite(rho):
                        rhos.append(rho)
                per_ratio.append({"score": score, "target": target, "ratio": ratio,
                                  "median_spearman": np.median(rhos) if rhos else np.nan,
                                  "blocks": len(rhos)})
    per_ratio = pd.DataFrame(per_ratio)
    per_ratio.to_csv(OUT / "train_vs_test_prediction_by_ratio.csv", index=False)
    if per_ratio.median_spearman.notna().any():
        print(per_ratio.pivot_table(index=["score", "ratio"], columns="target",
                                    values="median_spearman").to_string(
            float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT / 'train_vs_test_prediction.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="train_split_mae*.csv")
    parser.add_argument("--per-seed", action="store_true",
                        help="score individual selection seeds instead of their mean")
    return parser.parse_args()


if __name__ == "__main__":
    main()
