#!/usr/bin/env python3
"""Diagnose whether other sensors rescue sensor-local test novelty.

Each test atom is a (window, detector) pair.  Histories are made comparable
across detectors with detector-resolved fundamental-diagram scales:

    flow / capacity_flow,
    occupancy / critical_occupancy,
    speed / speed_at_capacity.

The audit separates three questions:

1. local support: is the history represented in the same detector's train set?
2. cross-sensor support: is it represented in another detector's train set?
3. conditional transferability: do those cross-sensor analogues have a future
   trajectory/state consistent with the realised target detector future?

Support itself uses nearest-support distance and therefore has no fixed-k
parameter.  Future consistency is reported at k=1,5,15 as a sensitivity curve;
the main interpretation never treats a single k as ground truth.  Finally the
diagnostic is linked to archived full-data errors from five forecasting models.

This is a retrospective empirical diagnostic, not a causal transfer estimate:
future consistency uses the realised future and is unavailable at prediction
time.  A targeted removal/retraining ablation is needed for causal evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fd_state_conditional_ambiguity import (  # noqa: E402
    observation_state,
    transition_codes,
    windows,
)
from fd_state_conditional_mae import (  # noqa: E402
    BACKBONE_DIRS,
    DATASETS,
    FITS_CSV,
    FULL_KEY,
    OUT,
    RANGE_END,
    REPO,
    STATE_NAMES,
    TEST_OFFSET,
    TEST_STEPS,
    T_IN,
    T_OUT,
    TRAIN_STEPS,
    leaf_map,
)


RNG_SEED = 20260721
K_FUTURE = (1, 5, 15)


def load_raw(dataset: str) -> np.memmap:
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    return np.memmap(
        REPO / "datasets" / dataset / "data.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(desc["shape"]),
    )


def fd_history(
    raw: np.memmap,
    sensor: int,
    starts: np.ndarray,
    capacity: float,
    critical_occupancy: float,
    speed_at_capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    physical = np.asarray(raw[:RANGE_END, sensor, :3], dtype=np.float32)
    scale = np.array(
        [capacity, critical_occupancy, speed_at_capacity], dtype=np.float32
    )
    scale = np.where(scale > 1e-8, scale, 1.0)
    history = windows(physical, starts, T_IN) / scale[None, None, :]
    valid = (
        np.isfinite(history).all(axis=(1, 2))
        & (history[:, :, 0] != 0).all(axis=1)
    )
    return history.reshape(len(starts), -1), valid


def fd_future_flow(
    raw: np.memmap,
    sensor: int,
    starts: np.ndarray,
    capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    flow = np.asarray(raw[:RANGE_END, sensor, 0], dtype=np.float32)
    future = windows(flow, starts + T_IN, T_OUT) / max(capacity, 1e-8)
    valid = np.isfinite(future).all(axis=1) & (future != 0).all(axis=1)
    return future, valid


def append_calendar(
    raw: np.memmap, starts: np.ndarray, feature: np.ndarray
) -> np.ndarray:
    """Append known-at-origin TOD/DOW using the same convention as STID."""
    origin = starts + T_IN - 1
    tod = np.floor(np.asarray(raw[origin, 0, 3]) * 288).astype(int)
    dow = np.floor(np.asarray(raw[origin, 0, 4]) * 7).astype(int)
    tod = np.clip(tod, 0, 287)
    dow = np.clip(dow, 0, 6)
    angle = 2 * np.pi * tod / 288
    calendar = np.concatenate([
        np.column_stack([np.sin(angle), np.cos(angle)]),
        np.eye(7, dtype=np.float32)[dow],
    ], axis=1).astype(np.float32)
    return np.concatenate([feature, calendar], axis=1)


def select_test_indices(
    code: np.ndarray, max_per_state: int, rng: np.random.Generator
) -> np.ndarray:
    selected = []
    for state in STATE_NAMES:
        index = np.flatnonzero(code == state)
        if len(index) > max_per_state:
            index = np.sort(rng.choice(index, max_per_state, replace=False))
        selected.append(index)
    return np.sort(np.concatenate(selected)) if selected else np.empty(0, int)


def coreset_seed(path: Path) -> int:
    text = (path.parent / "cfg.txt").read_text()
    block = text[text.find("CORESET:"):text.find("ENV:")]
    match = re.search(r"SEED:\s*(\d+)", block)
    return int(match.group(1)) if match else -1


def full_archive_path(paths: list[Path]) -> Path:
    seed42 = [path for path in paths if coreset_seed(path) == 42]
    candidates = seed42 or paths
    return max(candidates, key=lambda path: path.stat().st_mtime)


def prediction_errors(
    dataset: str,
    sensors: np.ndarray,
    query_window: np.ndarray,
    query_local_sensor: np.ndarray,
    query_target: np.ndarray,
    query_capacity: np.ndarray,
    backbones: list[str],
) -> dict[str, np.ndarray]:
    output = {}
    for backbone in backbones:
        leaves = leaf_map(BACKBONE_DIRS[backbone], dataset)
        paths = leaves.get(FULL_KEY, [])
        if not paths:
            warnings.warn(f"missing full archive: {dataset=} {backbone=}")
            continue
        path = full_archive_path(paths)
        with np.load(path) as archive:
            prediction = np.asarray(
                archive["prediction"][:, :, sensors, 0], dtype=np.float32
            )
        pred = prediction[
            query_window, :, query_local_sensor
        ] / query_capacity[:, None]
        valid = np.isfinite(query_target) & (query_target != 0)
        error = np.divide(
            np.where(valid, np.abs(pred - query_target), 0.0).sum(axis=1),
            valid.sum(axis=1),
            out=np.full(len(pred), np.nan),
            where=valid.sum(axis=1) > 0,
        )
        output[backbone] = error
        print(f"    loaded {backbone} full prediction", flush=True)
    return output


def foreign_neighbours(
    pool_embedding: np.ndarray,
    pool_sensor: np.ndarray,
    query_embedding: np.ndarray,
    query_sensor: np.ndarray,
    candidates: int,
    needed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_candidates = min(candidates, len(pool_embedding))
    nn = NearestNeighbors(
        n_neighbors=n_candidates,
        metric="euclidean",
        algorithm="kd_tree",
        leaf_size=80,
        n_jobs=n_jobs,
    ).fit(pool_embedding)
    distance, index = nn.kneighbors(query_embedding, return_distance=True)
    selected_index = np.full((len(query_embedding), needed), -1, dtype=np.int64)
    selected_distance = np.full((len(query_embedding), needed), np.nan)
    for row in range(len(query_embedding)):
        foreign = pool_sensor[index[row]] != query_sensor[row]
        valid_index = index[row, foreign][:needed]
        valid_distance = distance[row, foreign][:needed]
        selected_index[row, :len(valid_index)] = valid_index
        selected_distance[row, :len(valid_distance)] = valid_distance
    # A highly sensor-specific query can have only same-sensor atoms in the
    # global candidate list.  Resolve those rows exactly against a pool with
    # that sensor removed instead of silently increasing a global fixed k.
    incomplete = (selected_index < 0).any(axis=1)
    for sensor in np.unique(query_sensor[incomplete]):
        rows = np.flatnonzero(incomplete & (query_sensor == sensor))
        foreign_pool = np.flatnonzero(pool_sensor != sensor)
        fallback_k = min(needed, len(foreign_pool))
        fallback = NearestNeighbors(
            n_neighbors=fallback_k,
            metric="euclidean",
            algorithm="kd_tree",
            leaf_size=80,
            n_jobs=n_jobs,
        ).fit(pool_embedding[foreign_pool])
        d, i = fallback.kneighbors(query_embedding[rows], return_distance=True)
        selected_index[rows, :fallback_k] = foreign_pool[i]
        selected_distance[rows, :fallback_k] = d
    if (selected_index[:, 0] < 0).any():
        raise RuntimeError("No foreign-sensor training atom is available")
    return selected_distance, selected_index


def local_distances(
    raw: np.memmap,
    accepted: pd.DataFrame,
    query_raw_feature: np.ndarray,
    query_local_sensor: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    include_calendar: bool,
    n_jobs: int,
) -> np.ndarray:
    train_starts = np.arange(TRAIN_STEPS - T_IN - T_OUT + 1)
    output = np.full(len(query_raw_feature), np.nan)
    for local_sensor, row in accepted.iterrows():
        query_mask = query_local_sensor == local_sensor
        if not query_mask.any():
            continue
        feature, valid = fd_history(
            raw,
            int(row.sensor),
            train_starts,
            float(row.capacity_flow),
            float(row.critical_occupancy),
            float(row.speed_at_capacity),
        )
        if include_calendar:
            feature = append_calendar(raw, train_starts, feature)
        embedding = pca.transform(scaler.transform(feature[valid])).astype(np.float32)
        query_embedding = pca.transform(
            scaler.transform(query_raw_feature[query_mask])
        ).astype(np.float32)
        nn = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
            algorithm="kd_tree",
            n_jobs=n_jobs,
        ).fit(embedding)
        output[query_mask] = nn.kneighbors(
            query_embedding, return_distance=True
        )[0][:, 0]
    return output


def centred_correlation(
    frame: pd.DataFrame, metric: str, outcome: str
) -> tuple[float, float, int]:
    data = frame[[
        "dataset", "traffic_state_transition", metric, outcome
    ]].dropna().copy()
    groups = ["dataset", "traffic_state_transition"]
    data[metric] -= data.groupby(groups)[metric].transform("mean")
    data[outcome] -= data.groupby(groups)[outcome].transform("mean")
    if len(data) < 3 or data[metric].nunique() < 2 or data[outcome].nunique() < 2:
        return np.nan, np.nan, len(data)
    return (
        float(pearsonr(data[metric], data[outcome])[0]),
        float(spearmanr(data[metric], data[outcome])[0]),
        len(data),
    )


def summarise(atoms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # State-conditional novelty prevents transition states from being labelled
    # novel merely because they are rare relative to stable states.
    novelty_cut = atoms.groupby(
        ["dataset", "traffic_state_transition"]
    )["local_support_distance"].transform(lambda values: values.quantile(0.8))
    analog_cut = atoms.groupby(
        ["dataset", "traffic_state_transition"]
    )["cross_future_mae_k15"].transform("median")
    dispersion_cut = atoms.groupby(
        ["dataset", "traffic_state_transition"]
    )["cross_future_dispersion_k15"].transform("median")
    atoms["local_novel_q20"] = atoms.local_support_distance >= novelty_cut
    atoms["foreign_closer_than_local"] = atoms.transfer_advantage_log_ratio > 0
    atoms["cross_future_coherent"] = (
        atoms.cross_future_dispersion_k15 <= dispersion_cut
    )
    atoms["retrospective_future_consistent"] = (
        atoms.cross_future_mae_k15 <= analog_cut
    )
    atoms["exante_transfer_opportunity"] = (
        atoms.local_novel_q20
        & atoms.foreign_closer_than_local
        & atoms.cross_future_coherent
    )
    atoms["retrospective_verified_transfer"] = (
        atoms.local_novel_q20
        & atoms.foreign_closer_than_local
        & atoms.retrospective_future_consistent
    )
    atoms["support_group"] = "local_covered"
    atoms.loc[atoms.local_novel_q20, "support_group"] = "local_novel_no_rescue"
    atoms.loc[
        atoms.local_novel_q20 & atoms.foreign_closer_than_local,
        "support_group",
    ] = "local_novel_cross_ambiguous"
    atoms.loc[
        atoms.exante_transfer_opportunity,
        "support_group",
    ] = "local_novel_cross_coherent"

    metrics = [
        "model_mean_normalized_mae",
        "local_support_distance",
        "cross_support_distance",
        "transfer_advantage_log_ratio",
        "cross_future_mae_k1",
        "cross_future_mae_k5",
        "cross_future_mae_k15",
        "cross_future_dispersion_k5",
        "cross_future_dispersion_k15",
        "cross_state_match_rate_k15",
        "cross_state_purity_k15",
        "cross_sensor_diversity_k15",
    ]
    summary = (
        atoms.groupby(
            ["dataset", "traffic_state_transition", "support_group"],
            as_index=False,
        )
        .agg(
            n_atoms=("sensor", "size"),
            n_sensors=("sensor", "nunique"),
            **{metric: (metric, "mean") for metric in metrics},
        )
    )

    outcomes = [
        column for column in atoms.columns
        if column.endswith("_normalized_mae")
    ]
    diagnostic_metrics = [
        "local_support_distance",
        "cross_support_distance",
        "transfer_advantage_log_ratio",
        "cross_future_mae_k1",
        "cross_future_mae_k5",
        "cross_future_mae_k15",
        "cross_future_dispersion_k5",
        "cross_future_dispersion_k15",
        "cross_state_match_rate_k15",
        "cross_state_purity_k15",
    ]
    correlations = []
    for outcome in outcomes:
        for metric in diagnostic_metrics:
            pearson, spearman, n = centred_correlation(atoms, metric, outcome)
            correlations.append({
                "outcome": outcome,
                "diagnostic": metric,
                "n_atoms": n,
                "pearson_dataset_state_centered": pearson,
                "spearman_dataset_state_centered": spearman,
            })

    model_columns = [
        column for column in atoms.columns
        if column.endswith("_normalized_mae") and column != "model_mean_normalized_mae"
    ]
    by_model = atoms.melt(
        id_vars=["dataset", "traffic_state_transition", "support_group"],
        value_vars=model_columns,
        var_name="backbone",
        value_name="normalized_mae",
    )
    by_model["backbone"] = by_model.backbone.str.replace(
        "_normalized_mae", "", regex=False
    )
    model_summary = (
        by_model.groupby(
            ["dataset", "traffic_state_transition", "support_group", "backbone"],
            as_index=False,
        )
        .agg(normalized_mae=("normalized_mae", "mean"), n_atoms=("normalized_mae", "count"))
    )
    return summary, pd.DataFrame(correlations), model_summary


def cluster_bootstrap_contrasts(
    atoms: pd.DataFrame, reps: int = 2000
) -> pd.DataFrame:
    """Dataset-balanced contrasts with detector-cluster bootstrap intervals."""
    outcome = "model_mean_normalized_mae"
    aggregate = (
        atoms.groupby([
            "dataset", "sensor", "traffic_state_transition", "support_group"
        ])[outcome]
        .agg(["sum", "count"])
        .reset_index()
    )
    contrasts = [
        ("local_novel_cross_coherent", "local_novel_no_rescue"),
        ("local_novel_cross_coherent", "local_novel_cross_ambiguous"),
        ("local_novel_cross_coherent", "local_covered"),
    ]
    rng = np.random.default_rng(RNG_SEED + 8107)
    rows = []
    for state in STATE_NAMES.values():
        for left_group, right_group in contrasts:
            dataset_estimates = []
            dataset_draws = []
            for dataset in sorted(atoms.dataset.unique()):
                frame = atoms[
                    (atoms.dataset == dataset)
                    & (atoms.traffic_state_transition == state)
                ]
                left = frame[frame.support_group == left_group][outcome]
                right = frame[frame.support_group == right_group][outcome]
                dataset_estimates.append(left.mean() - right.mean())

                clustered = aggregate[
                    (aggregate.dataset == dataset)
                    & (aggregate.traffic_state_transition == state)
                ]
                sensors = np.sort(clustered.sensor.unique())
                weights = rng.multinomial(
                    len(sensors), np.full(len(sensors), 1 / len(sensors)), size=reps
                )
                group_draw = []
                for group in (left_group, right_group):
                    selected = clustered[clustered.support_group == group].set_index(
                        "sensor"
                    )
                    sums = selected["sum"].reindex(sensors, fill_value=0).to_numpy()
                    counts = selected["count"].reindex(sensors, fill_value=0).to_numpy()
                    numerator = weights @ sums
                    denominator = weights @ counts
                    group_draw.append(np.divide(
                        numerator,
                        denominator,
                        out=np.full(reps, np.nan),
                        where=denominator > 0,
                    ))
                dataset_draws.append(group_draw[0] - group_draw[1])
            estimate = float(np.nanmean(dataset_estimates))
            draws = np.nanmean(np.stack(dataset_draws), axis=0)
            low, high = np.nanquantile(draws, [0.025, 0.975])
            rows.append({
                "traffic_state_transition": state,
                "left_group": left_group,
                "right_group": right_group,
                "contrast_normalized_mae": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_unit": "detector within dataset",
                "bootstrap_reps": reps,
            })
    return pd.DataFrame(rows)


def analyse_dataset(dataset: str, args: argparse.Namespace) -> pd.DataFrame:
    fits = pd.read_csv(FITS_CSV)
    accepted = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor")
    accepted = accepted.reset_index(drop=True)
    if args.max_sensors is not None:
        accepted = accepted.head(args.max_sensors).reset_index(drop=True)
    sensors = accepted.sensor.to_numpy(int)
    raw = load_raw(dataset)
    train_starts = np.arange(0, TRAIN_STEPS - T_IN - T_OUT + 1, args.train_stride)
    test_starts = TEST_OFFSET + np.arange(TEST_STEPS - T_IN - T_OUT + 1)
    train_code = transition_codes(observation_state(raw[:TRAIN_STEPS], accepted))
    test_code = transition_codes(
        observation_state(raw[TEST_OFFSET:RANGE_END], accepted)
    )
    rng = np.random.default_rng(RNG_SEED + list(DATASETS).index(dataset) * 100_000)

    pool_feature = []
    pool_future = []
    pool_sensor = []
    pool_state = []
    pool_start = []
    query_feature = []
    query_future = []
    query_sensor = []
    query_local_sensor = []
    query_state = []
    query_window = []
    query_start = []
    query_capacity = []

    print(
        f"[{dataset}] sensors={len(accepted)} train_stride={args.train_stride}",
        flush=True,
    )
    for local_sensor, row in accepted.iterrows():
        history, history_valid = fd_history(
            raw,
            int(row.sensor),
            train_starts,
            float(row.capacity_flow),
            float(row.critical_occupancy),
            float(row.speed_at_capacity),
        )
        if args.include_calendar:
            history = append_calendar(raw, train_starts, history)
        future, future_valid = fd_future_flow(
            raw, int(row.sensor), train_starts, float(row.capacity_flow)
        )
        valid = history_valid & future_valid
        pool_feature.append(history[valid])
        pool_future.append(future[valid])
        pool_sensor.append(np.full(valid.sum(), int(row.sensor), dtype=np.int32))
        pool_state.append(train_code[train_starts[valid], local_sensor])
        pool_start.append(train_starts[valid])

        selected_window = select_test_indices(
            test_code[:, local_sensor], args.max_queries_per_state, rng
        )
        selected_start = test_starts[selected_window]
        history, history_valid = fd_history(
            raw,
            int(row.sensor),
            selected_start,
            float(row.capacity_flow),
            float(row.critical_occupancy),
            float(row.speed_at_capacity),
        )
        if args.include_calendar:
            history = append_calendar(raw, selected_start, history)
        future, future_valid = fd_future_flow(
            raw, int(row.sensor), selected_start, float(row.capacity_flow)
        )
        valid = history_valid & future_valid
        query_feature.append(history[valid])
        query_future.append(future[valid])
        query_sensor.append(np.full(valid.sum(), int(row.sensor), dtype=np.int32))
        query_local_sensor.append(
            np.full(valid.sum(), local_sensor, dtype=np.int16)
        )
        query_state.append(test_code[selected_window[valid], local_sensor])
        query_window.append(selected_window[valid])
        query_start.append(selected_start[valid])
        query_capacity.append(
            np.full(valid.sum(), float(row.capacity_flow), dtype=np.float32)
        )

    pool_feature = np.concatenate(pool_feature).astype(np.float32)
    pool_future = np.concatenate(pool_future).astype(np.float32)
    pool_sensor = np.concatenate(pool_sensor)
    pool_state = np.concatenate(pool_state)
    pool_start = np.concatenate(pool_start)
    query_feature = np.concatenate(query_feature).astype(np.float32)
    query_future = np.concatenate(query_future).astype(np.float32)
    query_sensor = np.concatenate(query_sensor)
    query_local_sensor = np.concatenate(query_local_sensor)
    query_state = np.concatenate(query_state)
    query_window = np.concatenate(query_window)
    query_start = np.concatenate(query_start)
    query_capacity = np.concatenate(query_capacity)
    print(
        f"  pool_atoms={len(pool_feature)} query_atoms={len(query_feature)}",
        flush=True,
    )

    scaler = StandardScaler().fit(pool_feature)
    pool_scaled = scaler.transform(pool_feature)
    pca = PCA(n_components=args.pca_dim, random_state=RNG_SEED)
    pool_embedding = pca.fit_transform(pool_scaled).astype(np.float32)
    query_embedding = pca.transform(scaler.transform(query_feature)).astype(np.float32)
    print(
        f"  PCA-{args.pca_dim} variance={pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )

    print("  computing exact within-sensor support", flush=True)
    local_distance = local_distances(
        raw,
        accepted,
        query_feature,
        query_local_sensor,
        scaler,
        pca,
        args.include_calendar,
        args.n_jobs,
    )
    print("  computing cross-sensor support and analogues", flush=True)
    cross_distance, cross_index = foreign_neighbours(
        pool_embedding,
        pool_sensor,
        query_embedding,
        query_sensor,
        args.cross_candidates,
        max(K_FUTURE),
        args.n_jobs,
    )
    neighbour_future = pool_future[cross_index]
    neighbour_state = pool_state[cross_index]
    neighbour_sensor = pool_sensor[cross_index]

    data = {
        "dataset": dataset,
        "sensor": query_sensor,
        "local_sensor": query_local_sensor,
        "test_window": query_window,
        "absolute_start": query_start,
        "traffic_state_transition": [STATE_NAMES[int(code)] for code in query_state],
        "local_support_distance": local_distance,
        "cross_support_distance": cross_distance[:, 0],
        "transfer_advantage_log_ratio": np.log(
            (local_distance + 1e-8) / (cross_distance[:, 0] + 1e-8)
        ),
        "cross_sensor_diversity_k15": np.array([
            len(np.unique(values)) for values in neighbour_sensor[:, :15]
        ]),
        "cross_state_match_rate_k15": (
            neighbour_state[:, :15] == query_state[:, None]
        ).mean(axis=1),
        "cross_state_purity_k15": np.array([
            max(np.bincount(values[values >= 0], minlength=4)) / len(values)
            if (values >= 0).any() else np.nan
            for values in neighbour_state[:, :15]
        ]),
    }
    for k in K_FUTURE:
        analog = np.median(neighbour_future[:, :k], axis=1)
        data[f"cross_future_mae_k{k}"] = np.abs(
            analog - query_future
        ).mean(axis=1)
        if k > 1:
            ensemble = neighbour_future[:, :k]
            pairwise = np.abs(
                ensemble[:, :, None, :] - ensemble[:, None, :, :]
            ).mean(axis=-1)
            data[f"cross_future_dispersion_k{k}"] = 0.5 * pairwise.mean(
                axis=(1, 2)
            )

    print("  loading full-data model errors", flush=True)
    error = prediction_errors(
        dataset,
        sensors,
        query_window,
        query_local_sensor,
        query_future,
        query_capacity,
        args.backbones,
    )
    for backbone, values in error.items():
        data[f"{backbone}_normalized_mae"] = values
    data["model_mean_normalized_mae"] = np.nanmean(
        np.stack(list(error.values())), axis=0
    )
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONE_DIRS), default=list(BACKBONE_DIRS))
    parser.add_argument("--max-sensors", type=int, default=None)
    parser.add_argument("--max-queries-per-state", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=12)
    parser.add_argument("--pca-dim", type=int, default=12)
    parser.add_argument("--cross-candidates", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--include-calendar", action="store_true",
        help="Append origin TOD (sin/cos) and DOW one-hot before PCA.",
    )
    parser.add_argument(
        "--output-tag", default="",
        help="Optional suffix for sensitivity-run CSVs (for example pca8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [analyse_dataset(dataset, args) for dataset in args.datasets]
    atoms = pd.concat(frames, ignore_index=True)
    summary, correlations, model_summary = summarise(atoms)
    contrasts = cluster_bootstrap_contrasts(atoms)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    atoms.to_csv(OUT / f"fd_cross_sensor_transfer_by_atom{suffix}.csv", index=False)
    summary.to_csv(OUT / f"fd_cross_sensor_transfer_summary{suffix}.csv", index=False)
    correlations.to_csv(
        OUT / f"fd_cross_sensor_transfer_correlations{suffix}.csv", index=False
    )
    model_summary.to_csv(
        OUT / f"fd_cross_sensor_transfer_by_model{suffix}.csv", index=False
    )
    contrasts.to_csv(
        OUT / f"fd_cross_sensor_transfer_bootstrap{suffix}.csv", index=False
    )

    print("\n=== support groups, pooled over datasets ===")
    display = (
        atoms.groupby(["traffic_state_transition", "support_group"])
        .agg(
            n=("sensor", "size"),
            sensors=("sensor", "nunique"),
            model_mae=("model_mean_normalized_mae", "mean"),
            local_d=("local_support_distance", "mean"),
            cross_d=("cross_support_distance", "mean"),
            analog_mae=("cross_future_mae_k15", "mean"),
            analog_dispersion=("cross_future_dispersion_k15", "mean"),
            state_match=("cross_state_match_rate_k15", "mean"),
        )
        .reset_index()
    )
    print(display.round(4).to_string(index=False))
    print("\n=== diagnostic versus five-model mean error ===")
    print(
        correlations[correlations.outcome == "model_mean_normalized_mae"]
        .round(3).to_string(index=False)
    )
    print("\nWrote fd_cross_sensor_transfer_{by_atom,summary,correlations,by_model}.csv")


if __name__ == "__main__":
    main()
