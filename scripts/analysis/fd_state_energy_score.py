#!/usr/bin/env python3
"""State-conditional trajectory Energy Score analysis.

This is a distributional counterpart to ``fd_state_conditional_ambiguity.py``.
For each detector and test history, the chronological training archive supplies
two empirical predictive distributions over the 12-step raw-flow trajectory:

    Q_X       futures of the k nearest resolved training histories (state blind)
    Q_XA      futures of the k nearest training histories in realised state A

Using rho(y, y') = ||y-y'||_2 / sqrt(H), the empirical Energy Score is

    ES(Q, y) = E_Q rho(Z, y) - 0.5 E_Q rho(Z, Z').

The distributional decomposition is computed directly as

    V = 0.5 E_{Q_XA} rho(Y, Y')
    G = 0.5 ED(Q_XA, Q_X) >= 0
    B = V + G,

where ED is the (biased V-statistic) energy distance.  Direct computation of G
is non-negative up to floating-point tolerance.  Observed test Energy Scores
under Q_X and Q_XA are also reported as an out-of-sample calibration check;
their finite-sample difference need not be non-negative.

The first-pass estimator deliberately retains the same kNN representation as
the existing ambiguity audit, so this script isolates what changes when the
point-MAE proxy is replaced by a proper trajectory-distribution score.  B/V
neighbour radii remain in the output to expose support mismatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fd_state_conditional_ambiguity import (  # noqa: E402
    DATASETS,
    FITS_CSV,
    OUT,
    RANGE_END,
    RNG_SEED,
    STATE_NAMES,
    TEST_OFFSET,
    TEST_STEPS,
    T_IN,
    T_OUT,
    TRAIN_STEPS,
    load_raw,
    median_pair_l1,
    observation_state,
    stratified_query_indices,
    transition_codes,
    windows,
)


PANEL_CSV = OUT / "fd_state_reducibility_multimodel.csv"
KEY = ["dataset", "sensor", "traffic_state_transition"]
METRICS = [
    "B_test_energy_score",
    "V_test_energy_score",
    "G_test_score_difference",
    "B_test_variogram_score_p05",
    "V_test_variogram_score_p05",
    "G_test_variogram_difference_p05",
    "B_distributional",
    "V_within_state_entropy",
    "G_energy_distance_half",
    "G_fraction_of_B",
    "B_calibration_gap",
    "V_calibration_gap",
    "decomposition_test_residual",
    "B_neighbor_radius",
    "V_neighbor_radius",
    "V_to_B_radius_ratio",
]


def trajectory_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean trajectory distance in per-horizon flow units."""
    delta = left[..., :, None, :] - right[..., None, :, :]
    return np.sqrt(np.mean(np.square(delta, dtype=np.float64), axis=-1))


def observed_energy_score(ensemble: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Energy score of equally weighted empirical ensembles, one per query."""
    target_distance = np.sqrt(
        np.mean(np.square(ensemble - target[:, None, :], dtype=np.float64), axis=-1)
    ).mean(axis=1)
    within = trajectory_distance(ensemble, ensemble).mean(axis=(1, 2))
    return target_distance - 0.5 * within


def observed_variogram_score(
    ensemble: np.ndarray, target: np.ndarray, p: float = 0.5
) -> np.ndarray:
    """Equal-weight variogram score over all unordered horizon pairs.

    The score complements the Energy Score by checking whether an empirical
    forecast distribution reproduces temporal increments, rather than only
    multivariate trajectory distances.  Equal pair weights avoid introducing
    an additional lag-weight hyperparameter.
    """
    ensemble_increment = np.abs(
        ensemble[..., :, None] - ensemble[..., None, :]
    ) ** p
    expected_increment = ensemble_increment.mean(axis=1)
    observed_increment = np.abs(
        target[..., :, None] - target[..., None, :]
    ) ** p
    upper = np.triu_indices(target.shape[-1], k=1)
    residual = observed_increment[:, upper[0], upper[1]] - expected_increment[
        :, upper[0], upper[1]
    ]
    return np.mean(np.square(residual, dtype=np.float64), axis=1)


def energy_decomposition(
    state_blind: np.ndarray, state_oracle: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return B, V, G using empirical score entropy and energy distance."""
    within_b = trajectory_distance(state_blind, state_blind).mean(axis=(1, 2))
    within_v = trajectory_distance(state_oracle, state_oracle).mean(axis=(1, 2))
    cross = trajectory_distance(state_blind, state_oracle).mean(axis=(1, 2))
    v = 0.5 * within_v
    g = cross - 0.5 * within_b - 0.5 * within_v
    # The biased energy-distance V-statistic is non-negative theoretically.
    # Only remove negligible numerical negatives; preserve larger violations as
    # an implementation diagnostic rather than silently changing them.
    g = np.where((g < 0) & (g > -1e-9), 0.0, g)
    b = v + g
    return b, v, g


def neighbour_archive(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k_max: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    effective_k = min(k_max, len(train_x))
    if effective_k < 2:
        return np.empty((len(query_x), 0, T_OUT)), np.empty((len(query_x), 0))
    nn = NearestNeighbors(
        n_neighbors=effective_k,
        metric="manhattan",
        algorithm="ball_tree",
        n_jobs=n_jobs,
    ).fit(train_x)
    distance, index = nn.kneighbors(query_x, return_distance=True)
    return train_y[index], distance


def scaled_history(
    raw: np.memmap,
    sensor: int,
    train_starts: np.ndarray,
    test_starts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    physical = np.asarray(raw[:RANGE_END, sensor, :3], np.float32)
    mean = physical[:TRAIN_STEPS].mean(axis=0)
    std = physical[:TRAIN_STEPS].std(axis=0)
    z = (physical - mean) / np.where(std > 0, std, 1.0)
    train_x = windows(z, train_starts, T_IN).reshape(len(train_starts), -1)
    test_x = windows(z, test_starts, T_IN).reshape(len(test_starts), -1)
    train_x = train_x / train_x.shape[1]
    test_x = test_x / test_x.shape[1]
    scale = median_pair_l1(train_x, rng)
    train_x, test_x = train_x / scale, test_x / scale
    flow = np.asarray(raw[:RANGE_END, sensor, 0], np.float32)
    train_y = windows(flow, train_starts + T_IN, T_OUT)
    test_y = windows(flow, test_starts + T_IN, T_OUT)
    return train_x, test_x, train_y, test_y


def detector_rows(
    dataset: str,
    sensor: int,
    local_sensor: int,
    raw: np.memmap,
    train_state: np.ndarray,
    test_state: np.ndarray,
    train_starts: np.ndarray,
    test_starts: np.ndarray,
    k_values: tuple[int, ...],
    max_queries_per_state: int | None,
    n_jobs: int,
) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED + 1009 * sensor + local_sensor)
    train_x, test_x, train_y, test_y = scaled_history(
        raw, sensor, train_starts, test_starts, rng
    )
    train_code = train_state[:, local_sensor]
    test_code = test_state[:, local_sensor]

    selected_query_index = stratified_query_indices(
        test_code, max_queries_per_state, rng
    )
    selected_count = {
        code: int((test_code[selected_query_index] == code).sum()) for code in STATE_NAMES
    }
    query_valid = (
        np.isfinite(test_y[selected_query_index]).all(axis=1)
        & (test_y[selected_query_index] != 0).all(axis=1)
    )
    query_index = selected_query_index[query_valid]
    if not len(query_index):
        return []

    # Complete trajectories keep rho a genuine common metric rather than a
    # query-dependent masked distance.  The accepted FD detectors have very
    # little missing flow, and exclusions are recorded below.
    train_y_valid = np.isfinite(train_y).all(axis=1) & (train_y != 0).all(axis=1)
    resolved = (train_code >= 0) & train_y_valid
    max_k = max(k_values)
    blind_y, blind_d = neighbour_archive(
        train_x[resolved], train_y[resolved], test_x[query_index], max_k, n_jobs
    )
    if blind_y.shape[1] < min(k_values):
        return []

    output: list[dict] = []
    query_code = test_code[query_index]
    query_y = test_y[query_index]
    for state_value, state_name in STATE_NAMES.items():
        qmask = query_code == state_value
        state_train = (train_code == state_value) & train_y_valid
        if not qmask.any() or state_train.sum() < min(k_values):
            continue
        oracle_y, oracle_d = neighbour_archive(
            train_x[state_train], train_y[state_train], test_x[query_index[qmask]], max_k, n_jobs
        )
        state_blind_y = blind_y[qmask]
        state_blind_d = blind_d[qmask]
        target = query_y[qmask]

        for requested_k in k_values:
            k = min(requested_k, state_blind_y.shape[1], oracle_y.shape[1])
            if k < 2:
                continue
            b_ens = state_blind_y[:, :k]
            v_ens = oracle_y[:, :k]
            b_test = observed_energy_score(b_ens, target)
            v_test = observed_energy_score(v_ens, target)
            b_variogram = observed_variogram_score(b_ens, target)
            v_variogram = observed_variogram_score(v_ens, target)
            b_dist, v_entropy, g_energy = energy_decomposition(b_ens, v_ens)
            if (g_energy < -1e-8).any():
                warnings.warn(
                    f"negative empirical energy distance: {dataset=} {sensor=} "
                    f"{state_name=} min={g_energy.min():.3e}"
                )
            b_radius = state_blind_d[:, k - 1]
            v_radius = oracle_d[:, k - 1]
            output.append({
                "dataset": dataset,
                "sensor": int(sensor),
                "traffic_state_transition": state_name,
                "k": int(requested_k),
                "n_train_resolved_valid": int(resolved.sum()),
                "n_train_state_valid": int(state_train.sum()),
                "n_test_state": int((test_code == state_value).sum()),
                "n_test_evaluated": int(qmask.sum()),
                "n_test_invalid_trajectory_excluded": int(
                    selected_count[state_value] - qmask.sum()
                ),
                "B_test_energy_score": float(b_test.mean()),
                "V_test_energy_score": float(v_test.mean()),
                "G_test_score_difference": float((b_test - v_test).mean()),
                "B_test_variogram_score_p05": float(b_variogram.mean()),
                "V_test_variogram_score_p05": float(v_variogram.mean()),
                "G_test_variogram_difference_p05": float(
                    (b_variogram - v_variogram).mean()
                ),
                "B_distributional": float(b_dist.mean()),
                "V_within_state_entropy": float(v_entropy.mean()),
                "G_energy_distance_half": float(g_energy.mean()),
                "G_fraction_of_B": float(np.divide(
                    g_energy.mean(), b_dist.mean(),
                    out=np.array(np.nan), where=b_dist.mean() > 0,
                )),
                "B_calibration_gap": float((b_test - b_dist).mean()),
                "V_calibration_gap": float((v_test - v_entropy).mean()),
                "decomposition_test_residual": float(
                    (b_test - v_test - g_energy).mean()
                ),
                "B_neighbor_radius": float(b_radius.mean()),
                "V_neighbor_radius": float(v_radius.mean()),
                "V_to_B_radius_ratio": float(
                    v_radius.mean() / b_radius.mean()
                    if b_radius.mean() > 0 else np.nan
                ),
            })
    return output


def bootstrap_summary(rows: pd.DataFrame, reps: int) -> pd.DataFrame:
    groups = ["k", "traffic_state_transition"]
    rng = np.random.default_rng(RNG_SEED + 1701)
    output = []
    for key, frame in rows.groupby(groups, sort=False):
        arrays = [d[METRICS].to_numpy(float) for _, d in frame.groupby("dataset")]
        estimate = np.nanmean(
            np.stack([np.nanmean(a, axis=0) for a in arrays]), axis=0
        )
        draws = np.empty((reps, len(METRICS)), float)
        for rep in range(reps):
            dataset_means = []
            for a in arrays:
                idx = rng.integers(0, len(a), len(a))
                dataset_means.append(np.nanmean(a[idx], axis=0))
            draws[rep] = np.nanmean(dataset_means, axis=0)
        low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
        row = dict(zip(groups, key))
        row["n_detector_rows"] = len(frame)
        for j, metric in enumerate(METRICS):
            row[metric] = estimate[j]
            row[f"{metric}_ci_low"] = low[j]
            row[f"{metric}_ci_high"] = high[j]
        output.append(row)
    return pd.DataFrame(output)


def dataset_center(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] -= result.groupby("dataset")[column].transform("mean")
    return result


def model_validation(rows: pd.DataFrame) -> pd.DataFrame:
    if not PANEL_CSV.exists():
        return pd.DataFrame()
    panel = pd.read_csv(PANEL_CSV)
    model = panel.groupby(KEY, as_index=False).agg(
        full_mae=("full_mae", "mean"),
        persist_mae=("persist_mae", "first"),
    )
    merged = rows.merge(model, on=KEY, how="inner")
    metrics = [
        "B_test_energy_score",
        "B_test_variogram_score_p05",
        "B_distributional",
        "V_within_state_entropy",
        "G_energy_distance_half",
    ]
    output = []
    for (k, state), frame in merged.groupby(["k", "traffic_state_transition"]):
        for metric in metrics:
            d = frame[["dataset", metric, "full_mae", "persist_mae"]].dropna()
            dc = dataset_center(d, [metric, "full_mae"])
            output.append({
                "k": int(k),
                "traffic_state_transition": state,
                "metric": metric,
                "n_detectors": len(dc),
                "pearson_dataset_centered": pearsonr(dc[metric], dc.full_mae)[0],
                "spearman_dataset_centered": spearmanr(dc[metric], dc.full_mae)[0],
            })
    return pd.DataFrame(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--k", nargs="+", type=int, default=(5, 15, 30))
    parser.add_argument("--max-queries-per-state", type=int, default=600)
    parser.add_argument("--max-detectors", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fits = pd.read_csv(FITS_CSV)
    train_starts = np.arange(TRAIN_STEPS - T_IN - T_OUT + 1)
    test_starts = TEST_OFFSET + np.arange(TEST_STEPS - T_IN - T_OUT + 1)
    all_rows = []
    for dataset in args.datasets:
        accepted = fits[(fits.dataset == dataset) & fits.identifiable].sort_values("sensor")
        if args.max_detectors is not None:
            accepted = accepted.head(args.max_detectors)
        raw = load_raw(dataset)
        train_state = transition_codes(observation_state(raw[:TRAIN_STEPS], accepted))
        test_state = transition_codes(
            observation_state(raw[TEST_OFFSET:RANGE_END], accepted)
        )
        print(f"[{dataset}] detectors={len(accepted)}", flush=True)
        for local_sensor, sensor in enumerate(accepted.sensor.to_numpy(int)):
            if local_sensor % 10 == 0:
                print(f"  detector {local_sensor}/{len(accepted)}", flush=True)
            all_rows.extend(detector_rows(
                dataset=dataset,
                sensor=int(sensor),
                local_sensor=local_sensor,
                raw=raw,
                train_state=train_state,
                test_state=test_state,
                train_starts=train_starts,
                test_starts=test_starts,
                k_values=tuple(sorted(set(args.k))),
                max_queries_per_state=args.max_queries_per_state,
                n_jobs=args.n_jobs,
            ))

    rows = pd.DataFrame(all_rows)
    if rows.empty:
        raise RuntimeError("No Energy Score rows were produced")
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "fd_state_energy_score_by_detector.csv", index=False)
    summary = bootstrap_summary(rows, args.bootstrap_reps)
    summary.to_csv(OUT / "fd_state_energy_score_summary.csv", index=False)
    validation = model_validation(rows)
    validation.to_csv(OUT / "fd_state_energy_score_model_validation.csv", index=False)

    main_k = min(args.k, key=lambda value: abs(value - 15))
    show = summary[summary.k == main_k][[
        "traffic_state_transition",
        "B_test_energy_score",
        "V_test_energy_score",
        "G_test_score_difference",
        "B_test_variogram_score_p05",
        "V_test_variogram_score_p05",
        "G_test_variogram_difference_p05",
        "B_distributional",
        "V_within_state_entropy",
        "G_energy_distance_half",
        "G_fraction_of_B",
        "V_to_B_radius_ratio",
    ]]
    print("\n=== k=15 state-conditional Energy Score ===")
    print(show.round(3).to_string(index=False))
    print("\n=== k=15 correlation with five-model full-data MAE ===")
    print(validation[validation.k == main_k].round(3).to_string(index=False))
    print("\nWrote fd_state_energy_score_{by_detector,summary,model_validation}.csv")


if __name__ == "__main__":
    main()
