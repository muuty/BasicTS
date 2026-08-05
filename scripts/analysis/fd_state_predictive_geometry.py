#!/usr/bin/env python3
"""State-conditional geometry of archived forecasting functions.

The phase-C archive does not store hidden activations, but it does store the
same test inputs and 12-step predictions for every backbone and coreset run.
This script therefore treats a model's 12-step forecast as a *predictive
function representation*.  It deliberately does not call this a hidden or
penultimate-layer representation.

For each traffic state it measures:

1. cross-backbone linear CKA among full-data predictive representations;
2. full-versus-coreset CKA, paired trajectory displacement, and a multiscale
   RBF-MMD approximation based on random Fourier features; and
3. association between those post-training geometry shifts and realised MAE
   degradation.

Trajectories are normalised by detector-specific flow scale estimated only on
the chronological training split.  Coreset predictions are averaged over one
run for each distinct selection seed (up to three); duplicate deterministic
reruns are collapsed by retaining the newest archive for that seed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from fd_state_conditional_mae import (
    BACKBONE_DIRS,
    DATASETS,
    FITS_CSV,
    FULL_KEY,
    OUT,
    REPO,
    STATE_NAMES,
    TRAIN_STEPS,
    leaf_map,
    test_transitions,
)


RNG_SEED = 20260721


def coreset_seed(path: Path) -> int:
    text = (path.parent / "cfg.txt").read_text()
    block = text[text.find("CORESET:"):text.find("ENV:")]
    match = re.search(r"SEED:\s*(\d+)", block)
    return int(match.group(1)) if match else -1


def distinct_seed_paths(paths: list[Path], max_seeds: int) -> list[Path]:
    """Collapse semantically duplicated reruns and retain distinct seeds."""
    newest: dict[int, Path] = {}
    for path in paths:
        seed = coreset_seed(path)
        if seed not in newest or path.stat().st_mtime > newest[seed].stat().st_mtime:
            newest[seed] = path
    return [newest[seed] for seed in sorted(newest)[:max_seeds]]


def mean_archive(
    paths: list[Path], sensors: np.ndarray, max_seeds: int, load_target: bool = True
) -> tuple[np.ndarray, Optional[np.ndarray], list[int]]:
    chosen = distinct_seed_paths(paths, max_seeds)
    if not chosen:
        raise ValueError("No archived predictions")
    total = None
    target = None
    for path in chosen:
        with np.load(path) as data:
            prediction = np.asarray(
                data["prediction"][:, :, sensors, 0], dtype=np.float32
            )
            if total is None:
                total = prediction.astype(np.float64)
                if load_target:
                    target = np.asarray(
                        data["target"][:, :, sensors, 0], dtype=np.float32
                    )
            else:
                total += prediction
    assert total is not None
    return (total / len(chosen)).astype(np.float32), target, [coreset_seed(p) for p in chosen]


def detector_scale(dataset: str, sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import json

    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    raw = np.memmap(
        REPO / "datasets" / dataset / "data.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(desc["shape"]),
    )
    train = np.asarray(raw[:TRAIN_STEPS, sensors, 0], dtype=np.float32)
    valid = np.isfinite(train) & (train != 0)
    count = valid.sum(axis=0)
    mean = np.divide(
        np.where(valid, train, 0).sum(axis=0),
        count,
        out=np.zeros(train.shape[1], dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(valid, train - mean[None, :], 0)
    variance = np.divide(
        np.square(centered, dtype=np.float64).sum(axis=0),
        count,
        out=np.ones(train.shape[1], dtype=np.float64),
        where=count > 0,
    )
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def atom_matrix(
    prediction: np.ndarray,
    atom_index: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    normalised = (prediction - mean[None, None, :]) / scale[None, None, :]
    return normalised.transpose(0, 2, 1).reshape(-1, prediction.shape[1])[atom_index]


def raw_atom_matrix(prediction: np.ndarray, atom_index: np.ndarray) -> np.ndarray:
    return prediction.transpose(0, 2, 1).reshape(-1, prediction.shape[1])[atom_index]


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    cross = np.square(left.T @ right).sum()
    left_norm = np.square(left.T @ left).sum()
    right_norm = np.square(right.T @ right).sum()
    denominator = np.sqrt(left_norm * right_norm)
    return float(cross / denominator) if denominator > 0 else np.nan


def median_bandwidth(values: np.ndarray, rng: np.random.Generator) -> float:
    n = len(values)
    draws = min(4096, max(n, 1))
    first = rng.integers(0, n, draws)
    second = rng.integers(0, n, draws)
    distance = np.sqrt(
        np.square(values[first] - values[second], dtype=np.float64).sum(axis=1)
    )
    positive = distance[distance > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def multiscale_rff_mmd(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    components_per_scale: int,
) -> tuple[float, float]:
    """Nonnegative RFF approximation to equal-weight multiscale RBF MMD."""
    sigma = median_bandwidth(left, rng)
    squared_mmd = []
    for multiplier in (0.5, 1.0, 2.0):
        bandwidth = max(sigma * multiplier, 1e-6)
        projection = rng.normal(
            0.0, 1.0 / bandwidth, size=(left.shape[1], components_per_scale)
        )
        phase = rng.uniform(0.0, 2.0 * np.pi, size=components_per_scale)
        scale = np.sqrt(2.0 / components_per_scale)
        left_mean = scale * np.cos(left @ projection + phase).mean(axis=0)
        right_mean = scale * np.cos(right @ projection + phase).mean(axis=0)
        squared_mmd.append(float(np.square(left_mean - right_mean).sum()))
    return float(np.sqrt(np.mean(squared_mmd))), sigma


def trajectory_mae(
    prediction: np.ndarray, target: np.ndarray, atom_index: np.ndarray
) -> float:
    pred = raw_atom_matrix(prediction, atom_index)
    truth = raw_atom_matrix(target, atom_index)
    valid = np.isfinite(truth) & (truth != 0)
    return float(np.abs(pred - truth)[valid].mean()) if valid.any() else np.nan


def sampled_atoms(
    code: np.ndarray, max_samples: int, rng: np.random.Generator
) -> dict[int, np.ndarray]:
    flattened = code.reshape(-1)
    result = {}
    for state in STATE_NAMES:
        index = np.flatnonzero(flattened == state)
        if len(index) > max_samples:
            index = np.sort(rng.choice(index, size=max_samples, replace=False))
        result[state] = index
    return result


def geometry_correlations(shift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "paired_prediction_shift_scaled_rms",
        "one_minus_full_reduced_cka",
        "multiscale_rbf_rff_mmd",
    ]
    for state, frame in shift.groupby("traffic_state_transition"):
        for metric in metrics:
            data = frame[["dataset", metric, "mae_degradation"]].dropna().copy()
            data[metric] -= data.groupby("dataset")[metric].transform("mean")
            data["mae_degradation"] -= data.groupby("dataset")[
                "mae_degradation"
            ].transform("mean")
            if len(data) < 2 or data[metric].nunique() < 2 or data[
                "mae_degradation"
            ].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = pearsonr(data[metric], data["mae_degradation"])[0]
                spearman = spearmanr(data[metric], data["mae_degradation"])[0]
            rows.append({
                "traffic_state_transition": state,
                "geometry_metric": metric,
                "n_cells": len(data),
                "pearson_dataset_centered": pearson,
                "spearman_dataset_centered": spearman,
            })
    for metric in metrics:
        data = shift[[
            "dataset", "traffic_state_transition", metric, "mae_degradation"
        ]].dropna().copy()
        groups = ["dataset", "traffic_state_transition"]
        data[metric] -= data.groupby(groups)[metric].transform("mean")
        data["mae_degradation"] -= data.groupby(groups)[
            "mae_degradation"
        ].transform("mean")
        rows.append({
            "traffic_state_transition": "all_within_dataset_state",
            "geometry_metric": metric,
            "n_cells": len(data),
            "pearson_dataset_centered": pearsonr(
                data[metric], data["mae_degradation"]
            )[0],
            "spearman_dataset_centered": spearmanr(
                data[metric], data["mae_degradation"]
            )[0],
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONE_DIRS), default=list(BACKBONE_DIRS))
    parser.add_argument(
        "--methods", nargs="+", default=["k_medoids", "k_center", "graph_cut", "random"]
    )
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--max-samples-per-state", type=int, default=2000)
    parser.add_argument("--rff-components-per-scale", type=int, default=64)
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fits = pd.read_csv(FITS_CSV)
    cka_rows = []
    shift_rows = []

    for dataset_index, dataset in enumerate(args.datasets):
        accepted = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor")
        accepted = accepted.reset_index(drop=True)
        min_valid = int(math.ceil(12 * args.min_valid_fraction))
        sensors, code = test_transitions(
            dataset, accepted, min_valid, args.min_agreement
        )
        mean, scale = detector_scale(dataset, sensors)
        sample_rng = np.random.default_rng(RNG_SEED + 1000 * dataset_index)
        state_atoms = sampled_atoms(code, args.max_samples_per_state, sample_rng)
        print(
            f"[{dataset}] detectors={len(sensors)} sampled="
            + str({STATE_NAMES[s]: len(v) for s, v in state_atoms.items()}),
            flush=True,
        )

        full_predictions: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        dataset_target = None
        leaves_by_backbone = {}
        for backbone in args.backbones:
            leaves = leaf_map(BACKBONE_DIRS[backbone], dataset)
            leaves_by_backbone[backbone] = leaves
            paths = leaves.get(FULL_KEY)
            if not paths:
                print(f"  {backbone}: missing full archive", flush=True)
                continue
            # All full-data archives use the same ENV seed and identical 100%
            # support; their nominal coreset seed has no effect on selection.
            prediction, target, seeds = mean_archive(
                paths, sensors, 1, load_target=dataset_target is None
            )
            if target is not None:
                dataset_target = target
            assert dataset_target is not None
            full_predictions[backbone] = prediction
            targets[backbone] = dataset_target
            print(f"  {backbone}: full selection seeds={seeds}", flush=True)

        # Cross-backbone agreement of the full-data forecasting functions.
        backbone_names = list(full_predictions)
        for state, state_name in STATE_NAMES.items():
            atoms = state_atoms[state]
            matrices = {
                backbone: atom_matrix(prediction, atoms, mean, scale)
                for backbone, prediction in full_predictions.items()
            }
            for left_pos, left_name in enumerate(backbone_names):
                for right_name in backbone_names[left_pos + 1:]:
                    cka_rows.append({
                        "dataset": dataset,
                        "traffic_state_transition": state_name,
                        "backbone_left": left_name,
                        "backbone_right": right_name,
                        "linear_cka": linear_cka(matrices[left_name], matrices[right_name]),
                        "n_atoms": len(atoms),
                    })

        # Geometry shift induced by each coreset-trained forecasting function.
        for backbone, full_prediction in full_predictions.items():
            leaves = leaves_by_backbone[backbone]
            target = targets[backbone]
            for method in args.methods:
                paths = leaves.get((method, args.ratio))
                if not paths:
                    print(f"  {backbone}: missing {method} r={args.ratio}", flush=True)
                    continue
                reduced_prediction, _, seeds = mean_archive(
                    paths, sensors, args.max_seeds, load_target=False
                )
                if reduced_prediction.shape != target.shape:
                    raise ValueError("Full and reduced test arrays have different shapes")
                for state, state_name in STATE_NAMES.items():
                    atoms = state_atoms[state]
                    full_matrix = atom_matrix(full_prediction, atoms, mean, scale)
                    reduced_matrix = atom_matrix(reduced_prediction, atoms, mean, scale)
                    difference = reduced_matrix - full_matrix
                    paired_shift = np.sqrt(
                        np.square(difference, dtype=np.float64).mean(axis=1)
                    ).mean()
                    seed_offset = (
                        RNG_SEED + dataset_index * 100_000
                        + list(BACKBONE_DIRS).index(backbone) * 10_000
                        + state
                    )
                    mmd, bandwidth = multiscale_rff_mmd(
                        full_matrix,
                        reduced_matrix,
                        np.random.default_rng(seed_offset),
                        args.rff_components_per_scale,
                    )
                    cka = linear_cka(full_matrix, reduced_matrix)
                    full_mae = trajectory_mae(full_prediction, target, atoms)
                    reduced_mae = trajectory_mae(reduced_prediction, target, atoms)
                    shift_rows.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "method": method,
                        "ratio": args.ratio,
                        "traffic_state_transition": state_name,
                        "n_atoms": len(atoms),
                        "selection_seeds": ",".join(map(str, seeds)),
                        "paired_prediction_shift_scaled_rms": float(paired_shift),
                        "full_reduced_linear_cka": cka,
                        "one_minus_full_reduced_cka": 1.0 - cka,
                        "multiscale_rbf_rff_mmd": mmd,
                        "rbf_median_bandwidth": bandwidth,
                        "full_mae": full_mae,
                        "reduced_mae": reduced_mae,
                        "mae_degradation": reduced_mae - full_mae,
                    })
                print(
                    f"  {backbone}: {method} r={args.ratio} seeds={seeds}", flush=True
                )

    OUT.mkdir(parents=True, exist_ok=True)
    cka = pd.DataFrame(cka_rows)
    shift = pd.DataFrame(shift_rows)
    validation = geometry_correlations(shift)
    cka.to_csv(OUT / "fd_state_predictive_geometry_cka.csv", index=False)
    shift.to_csv(OUT / "fd_state_predictive_geometry_shift.csv", index=False)
    validation.to_csv(
        OUT / "fd_state_predictive_geometry_validation.csv", index=False
    )

    print("\n=== full-data cross-backbone CKA ===")
    print(
        cka.groupby("traffic_state_transition")["linear_cka"]
        .agg(["mean", "min", "max"]).round(3).to_string()
    )
    print("\n=== full-versus-coreset predictive geometry shift ===")
    print(
        shift.groupby("traffic_state_transition")[[
            "paired_prediction_shift_scaled_rms",
            "one_minus_full_reduced_cka",
            "multiscale_rbf_rff_mmd",
            "mae_degradation",
        ]].mean().round(4).to_string()
    )
    print("\n=== geometry shift versus MAE degradation ===")
    print(validation.round(3).to_string(index=False))
    print("\nWrote fd_state_predictive_geometry_{cka,shift,validation}.csv")


if __name__ == "__main__":
    main()
