#!/usr/bin/env python3
"""State-conditional trajectory ambiguity with calendar conditioning.

This analysis estimates two different sources of state-conditional forecasting
difficulty without training a neural forecasting model.  For observable
conditioning variables U and a future-defined traffic transition A=a, let

    B_a(U) = E[MAE(Y, median(Y | U)) | A=a]
    V_a(U) = E[MAE(Y, median(Y | U, A=a)) | A=a]
    G_a(U) = B_a(U) - V_a(U).

``B`` is the error of a state-unaware conditional-median oracle, ``V`` is the
within-state trajectory variability left when the realised transition state is
also revealed, and ``G`` is the regime-identification penalty.  Since the true
conditional medians are unknown, this script reports leakage-free k-nearest-
neighbour proxies: the chronological training period is the reference archive
and the later test period is queried.  Neighbours are never drawn from the test
period.  The population quantity G is non-negative, but its finite-sample kNN
proxy need not be: rare-state neighbours are farther away than pooled-state
neighbours.  The reported B/V neighbour radii diagnose this support failure;
negative G estimates must not be interpreted as negative uncertainty.

Three observable conditioning sets are compared:

    X               standardised 12-step flow/occupancy/speed history
    X + TOD         X plus cyclic time-of-day
    X + TOD + DOW   X plus cyclic time-of-day and one-hot day-of-week

Each feature block is scaled by its median random-pair L1 distance in the
training archive.  ``--calendar-weights`` varies the relative calendar-block
weight; conclusions should not rely on a single value.  Multiple k values are
computed from one neighbour query.

Outputs
-------
fd_state_conditional_ambiguity_by_detector.csv
    Per dataset/detector/state/conditioning/weight/k estimates.
fd_state_conditional_ambiguity_summary.csv
    Equal-dataset detector means with stratified detector-bootstrap intervals.
fd_state_conditional_ambiguity_dataset_summary.csv
    Dataset-specific detector means.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
FITS_CSV = OUT / "fd_sensor_resolved_fits.csv"
DATASETS = ("SAN_BERNARDINO", "CONTRA_COSTA")
STATE_NAMES = {
    0: "free_to_free",
    1: "breakdown",
    2: "recovery",
    3: "congested_to_congested",
}

RANGE_END = 24_192
VALID_STEPS = int(RANGE_END * 0.2)
TEST_STEPS = int(RANGE_END * 0.2)
TRAIN_STEPS = RANGE_END - VALID_STEPS - TEST_STEPS
TEST_OFFSET = TRAIN_STEPS + VALID_STEPS
T_IN = T_OUT = 12
MIN_VALID = math.ceil(T_IN * 2 / 3)
MIN_AGREE = 0.75
STEPS_PER_DAY = 288
RNG_SEED = 20_260_720


def load_raw(dataset: str) -> np.memmap:
    root = REPO / "datasets" / "xtraffic" / dataset
    desc = json.loads((root / "desc.json").read_text())
    return np.memmap(
        root / "data.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(desc["shape"]),
    )


def observation_state(raw_segment: np.ndarray, accepted: pd.DataFrame) -> np.ndarray:
    sensors = accepted.sensor.to_numpy(int)
    occupancy = np.asarray(raw_segment[:, sensors, 1], np.float32)
    speed = np.asarray(raw_segment[:, sensors, 2], np.float32)
    critical_occupancy = accepted.critical_occupancy.to_numpy(float)[None, :]
    critical_speed = accepted.speed_at_capacity.to_numpy(float)[None, :]
    state = np.full(occupancy.shape, -1, np.int8)
    state[(occupancy < critical_occupancy) & (speed > critical_speed)] = 0
    state[(occupancy > critical_occupancy) & (speed < critical_speed)] = 1
    return state


def transition_codes(obs: np.ndarray) -> np.ndarray:
    """Traffic transition code for every complete 12+12 window in a segment."""
    free_prefix = np.vstack([
        np.zeros((1, obs.shape[1]), np.int32),
        (obs == 0).astype(np.int32).cumsum(axis=0),
    ])
    cong_prefix = np.vstack([
        np.zeros((1, obs.shape[1]), np.int32),
        (obs == 1).astype(np.int32).cumsum(axis=0),
    ])
    starts = np.arange(len(obs) - T_IN - T_OUT + 1)

    def segment_state(offset: int) -> np.ndarray:
        begin = starts + offset
        end = begin + T_IN
        n_free = free_prefix[end] - free_prefix[begin]
        n_cong = cong_prefix[end] - cong_prefix[begin]
        n_valid = n_free + n_cong
        state = np.full(n_valid.shape, -1, np.int8)
        valid = n_valid >= MIN_VALID
        state[valid & (n_free >= MIN_AGREE * n_valid)] = 0
        state[valid & (n_cong >= MIN_AGREE * n_valid)] = 1
        return state

    history, future = segment_state(0), segment_state(T_IN)
    code = np.full(history.shape, -1, np.int8)
    code[(history == 0) & (future == 0)] = 0
    code[(history == 0) & (future == 1)] = 1
    code[(history == 1) & (future == 0)] = 2
    code[(history == 1) & (future == 1)] = 3
    return code


def windows(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    return np.stack([values[starts + j] for j in range(length)], axis=1)


def calendar_blocks(raw: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calendar known at the last observed history step (same convention as STID)."""
    origin = starts + T_IN - 1
    # Calendar channels are replicated over detectors; use detector zero.
    tod = np.floor(np.asarray(raw[origin, 0, 3]) * STEPS_PER_DAY).astype(int)
    dow = np.floor(np.asarray(raw[origin, 0, 4]) * 7).astype(int)
    tod = np.clip(tod, 0, STEPS_PER_DAY - 1)
    dow = np.clip(dow, 0, 6)
    angle = 2 * np.pi * tod / STEPS_PER_DAY
    tod_block = np.column_stack([np.sin(angle), np.cos(angle)]).astype(np.float32)
    dow_block = np.eye(7, dtype=np.float32)[dow]
    return tod_block, dow_block


def median_pair_l1(block: np.ndarray, rng: np.random.Generator, n_pairs: int = 4096) -> float:
    left = rng.integers(0, len(block), n_pairs)
    right = rng.integers(0, len(block), n_pairs)
    distance = np.abs(block[left] - block[right]).sum(axis=1)
    positive = distance[distance > 1e-10]
    return float(np.median(positive)) if len(positive) else 1.0


def scaled_feature_sets(
    history_train: np.ndarray,
    history_test: np.ndarray,
    train_tod: np.ndarray,
    test_tod: np.ndarray,
    train_dow: np.ndarray,
    test_dow: np.ndarray,
    calendar_weights: tuple[float, ...],
    rng: np.random.Generator,
) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    # Mean-L1 history distance is used by the existing detector audit.  Divide
    # by dimension first, then median-normalise the block so the calendar
    # weight has an interpretable relative scale.
    h_train = history_train / history_train.shape[1]
    h_test = history_test / history_test.shape[1]
    h_scale = median_pair_l1(h_train, rng)
    t_scale = median_pair_l1(train_tod, rng)
    d_scale = median_pair_l1(train_dow, rng)
    h_train, h_test = h_train / h_scale, h_test / h_scale
    train_tod, test_tod = train_tod / t_scale, test_tod / t_scale
    train_dow, test_dow = train_dow / d_scale, test_dow / d_scale

    output = [("X", 0.0, h_train, h_test)]
    for weight in calendar_weights:
        output.append((
            "X+TOD",
            float(weight),
            np.concatenate([h_train, weight * train_tod], axis=1),
            np.concatenate([h_test, weight * test_tod], axis=1),
        ))
        output.append((
            "X+TOD+DOW",
            float(weight),
            np.concatenate([h_train, weight * train_tod, weight * train_dow], axis=1),
            np.concatenate([h_test, weight * test_tod, weight * test_dow], axis=1),
        ))
    return output


def stratified_query_indices(
    state: np.ndarray,
    max_per_state: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for code in STATE_NAMES:
        idx = np.flatnonzero(state == code)
        if max_per_state is not None and len(idx) > max_per_state:
            idx = np.sort(rng.choice(idx, max_per_state, replace=False))
        selected.append(idx)
    return np.sort(np.concatenate(selected)) if selected else np.array([], int)


def masked_trajectory_mae(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = np.isfinite(target) & (target != 0)
    numerator = np.where(valid, np.abs(prediction - target), 0.0).sum(axis=1)
    denominator = valid.sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(target), np.nan, dtype=float),
        where=denominator > 0,
    )


def neighbour_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k_values: tuple[int, ...],
    n_jobs: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    if len(train_x) == 0 or len(query_x) == 0:
        return {}, {}, np.empty((0, 0), dtype=int)
    k_max = min(max(k_values), len(train_x))
    nn = NearestNeighbors(
        n_neighbors=k_max,
        metric="manhattan",
        algorithm="ball_tree",
        n_jobs=n_jobs,
    ).fit(train_x)
    distance, index = nn.kneighbors(query_x, return_distance=True)
    prediction, radius = {}, {}
    neighbour_y = train_y[index]
    neighbour_y = np.where(neighbour_y == 0, np.nan, neighbour_y)
    for requested_k in k_values:
        k = min(requested_k, k_max)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            prediction[requested_k] = np.nanmedian(neighbour_y[:, :k], axis=1)
        radius[requested_k] = distance[:, k - 1]
    return prediction, radius, index


def detector_analysis(
    dataset: str,
    sensor: int,
    local_sensor: int,
    raw: np.memmap,
    train_code: np.ndarray,
    test_code: np.ndarray,
    train_starts: np.ndarray,
    test_starts_global: np.ndarray,
    train_tod: np.ndarray,
    test_tod: np.ndarray,
    train_dow: np.ndarray,
    test_dow: np.ndarray,
    calendar_weights: tuple[float, ...],
    k_values: tuple[int, ...],
    max_queries_per_state: int | None,
    n_jobs: int,
) -> list[dict]:
    physical = np.asarray(raw[:RANGE_END, sensor, :3], np.float32)
    mean = physical[:TRAIN_STEPS].mean(axis=0)
    std = physical[:TRAIN_STEPS].std(axis=0)
    z = (physical - mean) / np.where(std > 0, std, 1.0)
    history_train = windows(z, train_starts, T_IN).reshape(len(train_starts), -1)
    history_test = windows(z, test_starts_global, T_IN).reshape(len(test_starts_global), -1)
    flow = np.asarray(raw[:RANGE_END, sensor, 0], np.float32)
    future_train = windows(flow, train_starts + T_IN, T_OUT)
    future_test = windows(flow, test_starts_global + T_IN, T_OUT)
    state_train = train_code[:, local_sensor]
    state_test = test_code[:, local_sensor]

    sensor_rng = np.random.default_rng(RNG_SEED + 1009 * sensor + local_sensor)
    query_index = stratified_query_indices(
        state_test, max_queries_per_state, sensor_rng
    )
    if not len(query_index):
        return []
    state_query = state_test[query_index]
    y_query = future_test[query_index]

    feature_sets = scaled_feature_sets(
        history_train,
        history_test,
        train_tod,
        test_tod,
        train_dow,
        test_dow,
        calendar_weights,
        sensor_rng,
    )
    rows: list[dict] = []
    for feature_name, weight, x_train, x_test_all in feature_sets:
        x_query = x_test_all[query_index]
        # A is defined only on the four resolved transitions.  Use the same
        # resolved population for the state-unaware median and local transition
        # probabilities; unresolved reference atoms are outside this audit.
        resolved_train = state_train >= 0
        resolved_labels = state_train[resolved_train]
        b_pred, b_radius, b_index = neighbour_predictions(
            x_train[resolved_train], future_train[resolved_train], x_query, k_values, n_jobs
        )
        b_neighbour_state = resolved_labels[b_index]
        for state_code, state_name in STATE_NAMES.items():
            query_mask = state_query == state_code
            train_mask = state_train == state_code
            if not query_mask.any() or not train_mask.any():
                continue
            v_pred, v_radius, _ = neighbour_predictions(
                x_train[train_mask],
                future_train[train_mask],
                x_query[query_mask],
                k_values,
                n_jobs,
            )
            target = y_query[query_mask]
            for k in k_values:
                if k not in b_pred or k not in v_pred:
                    continue
                b_error = masked_trajectory_mae(b_pred[k][query_mask], target)
                v_error = masked_trajectory_mae(v_pred[k], target)
                valid = np.isfinite(b_error) & np.isfinite(v_error)
                if not valid.any():
                    continue
                b = float(b_error[valid].mean())
                v = float(v_error[valid].mean())
                g = b - v
                effective_k = min(k, b_neighbour_state.shape[1])
                local_labels = b_neighbour_state[query_mask, :effective_k][valid]
                alpha = 0.5
                denominator = effective_k + alpha * len(STATE_NAMES)
                class_counts = np.stack(
                    [(local_labels == code).sum(axis=1) for code in STATE_NAMES],
                    axis=1,
                )
                class_probability = (class_counts + alpha) / denominator
                realised_probability = class_probability[:, state_code]
                state_surprisal = -np.log(realised_probability)
                local_entropy = -np.sum(
                    class_probability * np.log(class_probability), axis=1
                )
                rows.append({
                    "dataset": dataset,
                    "sensor": int(sensor),
                    "traffic_state_transition": state_name,
                    "conditioning": feature_name,
                    "calendar_weight": weight,
                    "k": int(k),
                    "n_train_state": int(train_mask.sum()),
                    "n_test_state": int((state_test == state_code).sum()),
                    "n_test_evaluated": int(valid.sum()),
                    "B_state_unaware_mae": b,
                    "V_state_oracle_mae": v,
                    "G_regime_penalty": g,
                    "G_fraction_of_B": g / b if b > 0 else np.nan,
                    "realised_state_probability": float(realised_probability.mean()),
                    "state_surprisal_nats": float(state_surprisal.mean()),
                    "local_state_entropy_nats": float(local_entropy.mean()),
                    "realised_state_absent_rate": float(
                        (class_counts[:, state_code] == 0).mean()
                    ),
                    "B_neighbor_radius": float(np.mean(b_radius[k][query_mask][valid])),
                    "V_neighbor_radius": float(np.mean(v_radius[k][valid])),
                })
    return rows


def bootstrap_summary(rows: pd.DataFrame, reps: int) -> pd.DataFrame:
    metrics = [
        "B_state_unaware_mae",
        "V_state_oracle_mae",
        "G_regime_penalty",
        "G_fraction_of_B",
        "realised_state_probability",
        "state_surprisal_nats",
        "local_state_entropy_nats",
        "realised_state_absent_rate",
        "B_neighbor_radius",
        "V_neighbor_radius",
    ]
    groups = [
        "conditioning",
        "calendar_weight",
        "k",
        "traffic_state_transition",
    ]
    result = []
    rng = np.random.default_rng(RNG_SEED + 77)
    for key, frame in rows.groupby(groups, sort=False):
        dataset_arrays = [
            d[metrics].to_numpy(float)
            for _, d in frame.groupby("dataset", sort=False)
        ]
        estimate = np.nanmean(
            np.stack([np.nanmean(a, axis=0) for a in dataset_arrays]), axis=0
        )
        draws = np.empty((reps, len(metrics)), float)
        for rep in range(reps):
            dataset_means = []
            for array in dataset_arrays:
                idx = rng.integers(0, len(array), len(array))
                dataset_means.append(np.nanmean(array[idx], axis=0))
            draws[rep] = np.nanmean(np.stack(dataset_means), axis=0)
        low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
        row = dict(zip(groups, key))
        row["n_detectors"] = int(frame.sensor.nunique())
        row["n_detector_rows"] = int(len(frame))
        for idx, metric in enumerate(metrics):
            row[metric] = estimate[idx]
            row[f"{metric}_ci_low"] = low[idx]
            row[f"{metric}_ci_high"] = high[idx]
        result.append(row)
    return pd.DataFrame(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--calendar-weights", nargs="+", type=float, default=(0.5, 1.0))
    parser.add_argument("--k", nargs="+", type=int, default=(5, 15, 30))
    parser.add_argument("--max-queries-per-state", type=int, default=600)
    parser.add_argument("--max-detectors", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS_CSV)
    all_rows: list[dict] = []
    for dataset in args.datasets:
        accepted = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor")
        if args.max_detectors is not None:
            accepted = accepted.head(args.max_detectors)
        raw = load_raw(dataset)
        train_obs = observation_state(raw[:TRAIN_STEPS], accepted)
        test_obs = observation_state(raw[TEST_OFFSET:RANGE_END], accepted)
        train_code = transition_codes(train_obs)
        test_code = transition_codes(test_obs)
        train_starts = np.arange(TRAIN_STEPS - T_IN - T_OUT + 1)
        test_starts_global = TEST_OFFSET + np.arange(TEST_STEPS - T_IN - T_OUT + 1)
        train_tod, train_dow = calendar_blocks(raw, train_starts)
        test_tod, test_dow = calendar_blocks(raw, test_starts_global)
        print(
            f"[{dataset}] detectors={len(accepted)} "
            f"train_windows={len(train_starts)} test_windows={len(test_starts_global)}",
            flush=True,
        )
        for local_sensor, sensor in enumerate(accepted.sensor.to_numpy(int)):
            if local_sensor % 10 == 0:
                print(f"  detector {local_sensor}/{len(accepted)}", flush=True)
            all_rows.extend(detector_analysis(
                dataset=dataset,
                sensor=int(sensor),
                local_sensor=local_sensor,
                raw=raw,
                train_code=train_code,
                test_code=test_code,
                train_starts=train_starts,
                test_starts_global=test_starts_global,
                train_tod=train_tod,
                test_tod=test_tod,
                train_dow=train_dow,
                test_dow=test_dow,
                calendar_weights=tuple(args.calendar_weights),
                k_values=tuple(sorted(set(args.k))),
                max_queries_per_state=args.max_queries_per_state,
                n_jobs=args.n_jobs,
            ))
    rows = pd.DataFrame(all_rows)
    if rows.empty:
        raise RuntimeError("No state-conditional ambiguity rows were produced")
    rows.to_csv(OUT / "fd_state_conditional_ambiguity_by_detector.csv", index=False)
    dataset_summary = (
        rows.groupby([
            "dataset", "conditioning", "calendar_weight", "k",
            "traffic_state_transition",
        ], as_index=False)
        .agg(
            n_detectors=("sensor", "nunique"),
            B_state_unaware_mae=("B_state_unaware_mae", "mean"),
            V_state_oracle_mae=("V_state_oracle_mae", "mean"),
            G_regime_penalty=("G_regime_penalty", "mean"),
            G_fraction_of_B=("G_fraction_of_B", "mean"),
            realised_state_probability=("realised_state_probability", "mean"),
            state_surprisal_nats=("state_surprisal_nats", "mean"),
            local_state_entropy_nats=("local_state_entropy_nats", "mean"),
            realised_state_absent_rate=("realised_state_absent_rate", "mean"),
            B_neighbor_radius=("B_neighbor_radius", "mean"),
            V_neighbor_radius=("V_neighbor_radius", "mean"),
        )
    )
    dataset_summary.to_csv(
        OUT / "fd_state_conditional_ambiguity_dataset_summary.csv", index=False
    )
    summary = bootstrap_summary(rows, args.bootstrap_reps)
    summary.to_csv(OUT / "fd_state_conditional_ambiguity_summary.csv", index=False)

    main_weight = min(args.calendar_weights, key=lambda value: abs(value - 0.5))
    main_k = min(args.k, key=lambda value: abs(value - 15))
    show = summary[(summary.calendar_weight.isin([0.0, main_weight])) & (summary.k == main_k)]
    columns = [
        "conditioning", "traffic_state_transition", "B_state_unaware_mae",
        "V_state_oracle_mae", "G_regime_penalty", "G_fraction_of_B",
        "state_surprisal_nats",
    ]
    print("\n=== Main ambiguity estimates ===")
    print(show[columns].round(3).to_string(index=False))
    print("\nWrote fd_state_conditional_ambiguity_{by_detector,dataset_summary,summary}.csv")


if __name__ == "__main__":
    main()
