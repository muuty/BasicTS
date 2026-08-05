#!/usr/bin/env python3
"""Detector-clustered summaries for the sensor-resolved FD audit."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
SOURCE = OUT / "fd_sensor_resolved_by_detector.csv"
N_BOOT = 2000
METRICS = ["J_X", "future_gap", "residual", "residual_positive_rate", "D"]
STATES = [
    "free_to_free",
    "breakdown",
    "recovery",
    "congested_to_congested",
]


def stable_seed(values: tuple) -> int:
    return zlib.crc32("|".join(map(str, values)).encode())


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def average_selection_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "sensor", "method", "ratio",
        "traffic_state_transition", "K_label",
    ]
    return (
        rows.groupby(keys, as_index=False)
        .agg(
            n_selection_seeds=("selection_seed", "nunique"),
            n_atoms=("n_atoms", "first"),
            K=("K", "mean"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )


def clustered_summary(detector: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "method", "ratio", "traffic_state_transition", "K_label"
    ]
    output = []
    for group_key, group in detector.groupby(keys, sort=True):
        row = dict(zip(keys, group_key))
        row["n_detectors"] = int(group.sensor.nunique())
        row["n_atoms"] = int(group.n_atoms.sum())
        row["K_detector_mean"] = float(group.K.mean())
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_mean(values, stable_seed(group_key + (metric,)))
            row[f"{metric}_detector_mean"] = float(np.nanmean(values))
            row[f"{metric}_detector_median"] = float(np.nanmedian(values))
            row[f"{metric}_cluster_ci_low"] = lo
            row[f"{metric}_cluster_ci_high"] = hi
        output.append(row)
    return pd.DataFrame(output)


def paired_state_contrasts(detector: pd.DataFrame) -> pd.DataFrame:
    index = ["dataset", "sensor", "method", "ratio", "K_label"]
    output = []
    for metric in ["residual_positive_rate", "residual", "D", "future_gap", "J_X"]:
        wide = detector.pivot_table(
            index=index,
            columns="traffic_state_transition",
            values=metric,
            aggfunc="first",
        ).reset_index()
        if "free_to_free" not in wide:
            continue
        for state in STATES[1:]:
            if state not in wide:
                continue
            paired = wide.dropna(subset=["free_to_free", state]).copy()
            paired["difference"] = paired[state] - paired["free_to_free"]
            paired["ratio_to_free"] = np.divide(
                paired[state],
                paired["free_to_free"],
                out=np.full(len(paired), np.nan),
                where=paired["free_to_free"].to_numpy() > 1e-12,
            )
            group_keys = ["dataset", "method", "ratio", "K_label"]
            for group_key, group in paired.groupby(group_keys, sort=True):
                row = dict(zip(group_keys, group_key))
                row.update(metric=metric, compared_state=state)
                row["n_paired_detectors"] = int(len(group))
                for value_name in ["difference", "ratio_to_free"]:
                    values = group[value_name].to_numpy(dtype=float)
                    lo, hi = bootstrap_mean(
                        values,
                        stable_seed(group_key + (metric, state, value_name)),
                    )
                    row[f"{value_name}_mean"] = float(np.nanmean(values))
                    row[f"{value_name}_median"] = float(np.nanmedian(values))
                    row[f"{value_name}_ci_low"] = lo
                    row[f"{value_name}_ci_high"] = hi
                output.append(row)
    return pd.DataFrame(output)


def main() -> None:
    rows = pd.read_csv(SOURCE)
    detector = average_selection_seeds(rows)
    summary = clustered_summary(detector)
    contrasts = paired_state_contrasts(detector)
    detector.to_csv(OUT / "fd_sensor_resolved_seed_averaged.csv", index=False)
    summary.to_csv(OUT / "fd_sensor_resolved_clustered_summary.csv", index=False)
    contrasts.to_csv(OUT / "fd_sensor_resolved_state_contrasts.csv", index=False)
    print(
        f"wrote {len(detector)} detector rows, {len(summary)} summaries, "
        f"and {len(contrasts)} paired contrasts"
    )


if __name__ == "__main__":
    main()
