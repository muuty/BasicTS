#!/usr/bin/env python3
"""Detector-resolved fundamental-diagram support audit.

This script deliberately does not construct a network-average fundamental
diagram.  It fits a continuous two-branch flow--occupancy relation for every
detector on the chronological training split, confirms branch membership with
speed, leaves cross-quadrant observations unresolved, and matches each
detector's local history to the selected time support independently.

The output is a support-coverage diagnostic.  Detector-specific assignments
and weights are not the single reduced measure used to train a network model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
DATASETS = ("SAN_BERNARDINO", "CONTRA_COSTA")
METHODS = ("k_medoids", "random", "stride", "graph_cut",
           # deterministic constructions, written at r=0.1 under seed 42 only
           "fd_hyb0604", "fd_front10", "fd_front3g")
SEEDS = (42, 123, 456)
RATIOS = (0.1, 0.3)
T_IN = T_OUT = 12
RANGE_END = 24_192
VALID_STEPS = int(RANGE_END * 0.2)
TEST_STEPS = int(RANGE_END * 0.2)
TRAIN_STEPS = RANGE_END - VALID_STEPS - TEST_STEPS
N_WINDOWS = TRAIN_STEPS - T_IN - T_OUT + 1
DAY_STEPS = 24 * 12
STATE_NAMES = {
    0: "free_to_free",
    1: "breakdown",
    2: "recovery",
    3: "congested_to_congested",
}


@dataclass
class FDScreen:
    sensor: int
    identifiable: bool
    reason: str
    n_bins: int
    critical_bin: int
    critical_occupancy: float
    capacity_flow: float
    speed_at_capacity: float
    positive_speed_fraction: float
    free_slope: float
    congested_slope: float
    free_rise: float
    congested_drop: float
    piecewise_sse: float
    linear_sse: float
    fit_improvement: float
    piecewise_r2: float


def load_raw(dataset: str) -> np.ndarray:
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    data = np.memmap(
        REPO / "datasets" / dataset / "data.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(desc["shape"]),
    )
    return np.asarray(data[:TRAIN_STEPS, :, :3], dtype=np.float32)


def binned_detector(raw: np.ndarray, sensor: int, n_bins: int) -> pd.DataFrame:
    q = raw[:, sensor, 0]
    o = raw[:, sensor, 1]
    v = raw[:, sensor, 2]
    finite = np.isfinite(q) & np.isfinite(o) & np.isfinite(v)
    order = np.flatnonzero(finite)[np.argsort(o[finite], kind="mergesort")]
    bins = np.array_split(order, n_bins)
    return pd.DataFrame({
        "occupancy": [float(np.median(o[idx])) for idx in bins],
        "flow": [float(np.median(q[idx])) for idx in bins],
        "speed": [float(np.median(v[idx])) for idx in bins],
        "n": [int(len(idx)) for idx in bins],
    })


def fit_detector_fd(
    raw: np.ndarray,
    sensor: int,
    n_bins: int = 40,
    min_branch_bins: int = 6,
    min_free_rise: float = 0.10,
    min_congested_drop: float = 0.05,
    min_fit_improvement: float = 0.10,
) -> tuple[FDScreen, pd.DataFrame]:
    """Fit a constrained continuous two-branch relation to binned medians."""
    bins = binned_detector(raw, sensor, n_bins)
    o = bins["occupancy"].to_numpy(dtype=float)
    q = bins["flow"].to_numpy(dtype=float)

    linear_x = np.column_stack([np.ones_like(o), o])
    linear_beta, *_ = np.linalg.lstsq(linear_x, q, rcond=None)
    linear_resid = q - linear_x @ linear_beta
    linear_sse = float(linear_resid @ linear_resid)
    total_sse = float(((q - q.mean()) ** 2).sum())

    best = None
    for critical_bin in range(min_branch_bins - 1, n_bins - min_branch_bins):
        oc = float(o[critical_bin])
        left = np.minimum(o - oc, 0.0)
        right = np.maximum(o - oc, 0.0)
        design = np.column_stack([np.ones_like(o), left, right])
        fit = lsq_linear(
            design,
            q,
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, 0.0]),
            method="trf",
        )
        resid = q - design @ fit.x
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, critical_bin, oc, fit.x)

    assert best is not None
    piecewise_sse, critical_bin, oc, beta = best
    qc, free_slope, congested_slope = map(float, beta)
    q_left = qc + free_slope * float(o.min() - oc)
    q_right = qc + congested_slope * float(o.max() - oc)
    scale = max(abs(qc), 1e-8)
    free_rise = float((qc - q_left) / scale)
    congested_drop = float((qc - q_right) / scale)
    fit_improvement = float(
        (linear_sse - piecewise_sse) / max(linear_sse, 1e-8)
    )
    piecewise_r2 = float(1.0 - piecewise_sse / max(total_sse, 1e-8))
    speed_at_capacity = float(bins.loc[critical_bin, "speed"])
    positive_speed_fraction = float((raw[:, sensor, 2] > 0).mean())

    checks = [
        (critical_bin >= min_branch_bins - 1, "insufficient_free_support"),
        (critical_bin <= n_bins - min_branch_bins - 1, "insufficient_congested_support"),
        (free_slope > 1e-10, "non_increasing_free_branch"),
        (congested_slope < -1e-10, "non_decreasing_congested_branch"),
        (free_rise >= min_free_rise, "weak_free_branch"),
        (congested_drop >= min_congested_drop, "weak_congested_branch"),
        (fit_improvement >= min_fit_improvement, "weak_two_branch_improvement"),
        (positive_speed_fraction >= 0.99, "invalid_speed_channel"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    screen = FDScreen(
        sensor=sensor,
        identifiable=not failures,
        reason="accepted" if not failures else ";".join(failures),
        n_bins=n_bins,
        critical_bin=int(critical_bin),
        critical_occupancy=oc,
        capacity_flow=qc,
        speed_at_capacity=speed_at_capacity,
        positive_speed_fraction=positive_speed_fraction,
        free_slope=free_slope,
        congested_slope=congested_slope,
        free_rise=free_rise,
        congested_drop=congested_drop,
        piecewise_sse=piecewise_sse,
        linear_sse=linear_sse,
        fit_improvement=fit_improvement,
        piecewise_r2=piecewise_r2,
    )
    bins["sensor"] = sensor
    bins["fitted_flow"] = (
        qc
        + free_slope * np.minimum(o - oc, 0.0)
        + congested_slope * np.maximum(o - oc, 0.0)
    )
    bins["critical_bin"] = critical_bin
    bins["identifiable"] = screen.identifiable
    return screen, bins


def fit_all_detectors(
    raw: np.ndarray,
    n_bins: int,
    min_branch_bins: int,
    min_free_rise: float,
    min_congested_drop: float,
    min_fit_improvement: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    screens, bins = [], []
    for sensor in range(raw.shape[1]):
        screen, sensor_bins = fit_detector_fd(
            raw,
            sensor,
            n_bins=n_bins,
            min_branch_bins=min_branch_bins,
            min_free_rise=min_free_rise,
            min_congested_drop=min_congested_drop,
            min_fit_improvement=min_fit_improvement,
        )
        screens.append(asdict(screen))
        bins.append(sensor_bins)
    return pd.DataFrame(screens), pd.concat(bins, ignore_index=True)


def classify_observations(raw: np.ndarray, accepted: pd.DataFrame) -> np.ndarray:
    """Return -1 unresolved, 0 free, 1 congested for retained detectors."""
    sensors = accepted["sensor"].to_numpy(dtype=int)
    o = raw[:, sensors, 1]
    v = raw[:, sensors, 2]
    oc = accepted["critical_occupancy"].to_numpy(dtype=float)[None, :]
    vc = accepted["speed_at_capacity"].to_numpy(dtype=float)[None, :]
    state = np.full(o.shape, -1, dtype=np.int8)
    state[(o < oc) & (v > vc)] = 0
    state[(o > oc) & (v < vc)] = 1
    return state


def horizon_state(
    observation_state: np.ndarray,
    starts: np.ndarray,
    min_valid_fraction: float,
    min_agreement: float,
) -> np.ndarray:
    min_valid = int(math.ceil(T_IN * min_valid_fraction))
    free = (observation_state == 0).astype(np.int16)
    congested = (observation_state == 1).astype(np.int16)
    free_prefix = np.vstack([np.zeros((1, free.shape[1]), np.int32), free.cumsum(0)])
    cong_prefix = np.vstack([
        np.zeros((1, congested.shape[1]), np.int32), congested.cumsum(0)
    ])
    stop = starts + T_IN
    n_free = free_prefix[stop] - free_prefix[starts]
    n_cong = cong_prefix[stop] - cong_prefix[starts]
    n_valid = n_free + n_cong
    state = np.full(n_valid.shape, -1, dtype=np.int8)
    valid = n_valid >= min_valid
    state[valid & (n_free >= min_agreement * n_valid)] = 0
    state[valid & (n_cong >= min_agreement * n_valid)] = 1
    return state


def transition_codes(
    observation_state: np.ndarray,
    min_valid_fraction: float,
    min_agreement: float,
) -> np.ndarray:
    starts = np.arange(N_WINDOWS)
    history = horizon_state(
        observation_state, starts, min_valid_fraction, min_agreement
    )
    future = horizon_state(
        observation_state, starts + T_IN, min_valid_fraction, min_agreement
    )
    code = np.full(history.shape, -1, dtype=np.int8)
    code[(history == 0) & (future == 0)] = 0
    code[(history == 0) & (future == 1)] = 1
    code[(history == 1) & (future == 0)] = 2
    code[(history == 1) & (future == 1)] = 3
    return code


def standardise_per_detector(raw: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    local = raw[:, sensors, :].astype(np.float32, copy=True)
    mean = local.mean(axis=0, keepdims=True)
    std = local.std(axis=0, keepdims=True)
    return (local - mean) / np.where(std > 0, std, 1.0)


def local_windows(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Materialise [sensor, window, 36] history and future arrays."""
    offsets = np.arange(T_IN)[None, :]
    starts = np.arange(N_WINDOWS)[:, None]
    history = z[starts + offsets].transpose(2, 0, 1, 3).reshape(
        z.shape[1], N_WINDOWS, -1
    )
    future = z[starts + T_IN + offsets].transpose(2, 0, 1, 3).reshape(
        z.shape[1], N_WINDOWS, -1
    )
    return history.astype(np.float32), future.astype(np.float32)


def calendar_k_curve(history: np.ndarray, future: np.ndarray) -> dict[str, np.ndarray]:
    """Per-detector K curve from selector-independent calendar-lag pairs.

    Pairing the same clock time at lags of one through seven days avoids using
    any candidate support.  Multiple lags are used because a detector can be
    exactly weekly periodic after upstream data processing, which would make a
    one-week-only slope undefined.
    """
    slope_blocks = []
    for days in range(1, 8):
        lag = days * DAY_STEPS
        dx = np.abs(history[:, lag:] - history[:, :-lag]).mean(axis=2)
        dy = np.abs(future[:, lag:] - future[:, :-lag]).mean(axis=2)
        slope_blocks.append(np.divide(
            dy, dx, out=np.full_like(dy, np.nan), where=dx > 1e-8
        ))
    slopes = np.concatenate(slope_blocks, axis=1)
    if np.isnan(slopes).all(axis=1).any():
        raise ValueError("A detector has no non-zero calendar-lag history pairs")
    return {
        "K0": np.zeros(history.shape[0], dtype=np.float32),
        "K50": np.nanquantile(slopes, 0.50, axis=1).astype(np.float32),
        "K75": np.nanquantile(slopes, 0.75, axis=1).astype(np.float32),
        "K90": np.nanquantile(slopes, 0.90, axis=1).astype(np.float32),
        "K95": np.nanquantile(slopes, 0.95, axis=1).astype(np.float32),
    }


def load_indices(dataset: str, method: str, ratio: float, seed: int) -> np.ndarray:
    path = (
        REPO
        / "coreset_indices"
        / dataset
        / f"{method}_euclidean_{int(round(100 * ratio)):03d}_seed{seed}.json"
    )
    selected = np.asarray(json.loads(path.read_text()), dtype=np.int64)
    if selected.min() < 0 or selected.max() >= N_WINDOWS:
        raise ValueError(f"Out-of-range selected index in {path}")
    return selected


def nearest_local_support(
    history: np.ndarray,
    future: np.ndarray,
    selected: np.ndarray,
    device: str,
    query_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean-L1 nearest support for a batch of detectors."""
    import torch

    h = torch.as_tensor(history, device=device)
    f = torch.as_tensor(future, device=device)
    support = h[:, selected, :]
    n_sensors, n_windows, dimension = h.shape
    dx = torch.empty((n_sensors, n_windows), dtype=h.dtype, device=device)
    representative = torch.empty(
        (n_sensors, n_windows), dtype=torch.long, device=device
    )
    selected_t = torch.as_tensor(selected, dtype=torch.long, device=device)
    for begin in range(0, n_windows, query_batch):
        end = min(begin + query_batch, n_windows)
        dist = torch.cdist(h[:, begin:end, :], support, p=1) / dimension
        values, locations = dist.min(dim=2)
        dx[:, begin:end] = values
        representative[:, begin:end] = selected_t[locations]
    matched_future = torch.gather(
        f,
        1,
        representative[:, :, None].expand(-1, -1, dimension),
    )
    dy = torch.abs(f - matched_future).mean(dim=2)
    return dx.cpu().numpy(), dy.cpu().numpy()


def nearest_local_support_many(
    history: np.ndarray,
    future: np.ndarray,
    selected_sets: list[np.ndarray],
    device: str,
    query_batch: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact assignments for many supports using one distance-to-union pass."""
    import torch

    h = torch.as_tensor(history, device=device)
    f = torch.as_tensor(future, device=device)
    n_sensors, n_windows, dimension = h.shape
    union = np.unique(np.concatenate(selected_sets))
    position = np.full(n_windows, -1, dtype=np.int64)
    position[union] = np.arange(len(union))
    support_positions = [
        torch.as_tensor(position[selected], dtype=torch.long, device=device)
        for selected in selected_sets
    ]
    selected_tensors = [
        torch.as_tensor(selected, dtype=torch.long, device=device)
        for selected in selected_sets
    ]
    union_support = h[:, union, :]
    dx_all = [
        torch.empty((n_sensors, n_windows), dtype=h.dtype, device=device)
        for _ in selected_sets
    ]
    representative_all = [
        torch.empty((n_sensors, n_windows), dtype=torch.long, device=device)
        for _ in selected_sets
    ]
    for begin in range(0, n_windows, query_batch):
        end = min(begin + query_batch, n_windows)
        dist_union = torch.cdist(
            h[:, begin:end, :], union_support, p=1
        ) / dimension
        for config_i, (support_pos, selected_t) in enumerate(
            zip(support_positions, selected_tensors)
        ):
            values, locations = dist_union.index_select(2, support_pos).min(dim=2)
            dx_all[config_i][:, begin:end] = values
            representative_all[config_i][:, begin:end] = selected_t[locations]

    output = []
    for dx, representative in zip(dx_all, representative_all):
        matched_future = torch.gather(
            f,
            1,
            representative[:, :, None].expand(-1, -1, dimension),
        )
        dy = torch.abs(f - matched_future).mean(dim=2)
        output.append((dx.cpu().numpy(), dy.cpu().numpy()))
    return output


def detector_rows(
    dataset: str,
    sensor_ids: np.ndarray,
    transition: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    k_curve: dict[str, np.ndarray],
    method: str,
    ratio: float,
    seed: int,
) -> Iterable[dict]:
    # arrays use [sensor, window], whereas transition is [window, sensor]
    transition = transition.T
    for local_s, sensor in enumerate(sensor_ids):
        for state_code, state_name in STATE_NAMES.items():
            mask = transition[local_s] == state_code
            if not mask.any():
                continue
            state_dx = dx[local_s, mask]
            state_dy = dy[local_s, mask]
            for k_label, k_by_sensor in k_curve.items():
                k = float(k_by_sensor[local_s])
                residual = np.maximum(state_dy - k * state_dx, 0.0)
                yield {
                    "dataset": dataset,
                    "sensor": int(sensor),
                    "method": method,
                    "ratio": ratio,
                    "selection_seed": seed,
                    "traffic_state_transition": state_name,
                    "K_label": k_label,
                    "K": k,
                    "n_atoms": int(mask.sum()),
                    "J_X": float(state_dx.mean()),
                    "future_gap": float(state_dy.mean()),
                    "residual": float(residual.mean()),
                    "residual_positive_rate": float((residual > 0).mean()),
                    "D": float((k * state_dx + residual).mean()),
                }


def analyse_support(
    dataset: str,
    raw: np.ndarray,
    accepted: pd.DataFrame,
    transition: np.ndarray,
    methods: tuple[str, ...],
    ratios: tuple[float, ...],
    seeds: tuple[int, ...],
    device: str,
    sensor_batch: int,
    query_batch: int,
) -> pd.DataFrame:
    sensors = accepted["sensor"].to_numpy(dtype=int)
    all_rows: list[dict] = []
    # Several archived selectors are deterministic even though files exist for
    # three seed labels.  Group byte-identical selected sets so their expensive
    # detector-level nearest assignment is computed once, while retaining one
    # output row per declared seed for provenance.
    config_groups: dict[tuple[str, float, bytes], dict] = {}
    for method in methods:
        for ratio in ratios:
            for seed in seeds:
                selected = load_indices(dataset, method, ratio, seed)
                key = (method, ratio, selected.tobytes())
                if key not in config_groups:
                    config_groups[key] = {
                        "method": method,
                        "ratio": ratio,
                        "selected": selected,
                        "seeds": [],
                    }
                config_groups[key]["seeds"].append(seed)
    for begin in range(0, len(sensors), sensor_batch):
        end = min(begin + sensor_batch, len(sensors))
        sensor_ids = sensors[begin:end]
        print(f"  detector batch {begin}:{end}/{len(sensors)}", flush=True)
        z = standardise_per_detector(raw, sensor_ids)
        history, future = local_windows(z)
        k_curve = calendar_k_curve(history, future)
        local_transition = transition[:, begin:end]
        configs = list(config_groups.values())
        assignments = nearest_local_support_many(
            history,
            future,
            [config["selected"] for config in configs],
            device=device,
            query_batch=query_batch,
        )
        for config, (dx, dy) in zip(configs, assignments):
            method = config["method"]
            ratio = config["ratio"]
            for seed in config["seeds"]:
                all_rows.extend(detector_rows(
                    dataset,
                    sensor_ids,
                    local_transition,
                    dx,
                    dy,
                    k_curve,
                    method,
                    ratio,
                    seed,
                ))
            print(
                f"    {method:10s} r={ratio:.1f} seeds={config['seeds']} "
                f"Jx={dx.mean():.4f}",
                flush=True,
            )
    return pd.DataFrame(all_rows)


def summarise_detector_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group = [
        "dataset",
        "method",
        "ratio",
        "selection_seed",
        "traffic_state_transition",
        "K_label",
    ]
    return (
        rows.groupby(group, as_index=False)
        .agg(
            n_detectors=("sensor", "nunique"),
            n_atoms=("n_atoms", "sum"),
            K_detector_mean=("K", "mean"),
            J_X_detector_mean=("J_X", "mean"),
            future_gap_detector_mean=("future_gap", "mean"),
            residual_detector_mean=("residual", "mean"),
            positive_rate_detector_mean=("residual_positive_rate", "mean"),
            D_detector_mean=("D", "mean"),
            D_detector_median=("D", "median"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--min-branch-bins", type=int, default=6)
    parser.add_argument("--min-free-rise", type=float, default=0.10)
    parser.add_argument("--min-congested-drop", type=float, default=0.05)
    parser.add_argument("--min-fit-improvement", type=float, default=0.10)
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    parser.add_argument("--out-suffix", default="",
                        help="appended to every output filename, so a run over a "
                             "different method set does not overwrite an earlier one")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument(
        "--max-detectors",
        type=int,
        default=None,
        help="Debug-only cap applied after screening; omit for paper analysis.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sensor-batch", type=int, default=8)
    parser.add_argument("--query-batch", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_screens, all_bins, all_counts, all_rows = [], [], [], []
    for dataset in args.datasets:
        print(f"[{dataset}] detector-level FD fitting", flush=True)
        raw = load_raw(dataset)
        screens, bins = fit_all_detectors(
            raw,
            n_bins=args.n_bins,
            min_branch_bins=args.min_branch_bins,
            min_free_rise=args.min_free_rise,
            min_congested_drop=args.min_congested_drop,
            min_fit_improvement=args.min_fit_improvement,
        )
        screens["dataset"] = dataset
        bins["dataset"] = dataset
        accepted = screens[screens["identifiable"]].sort_values("sensor")
        print(
            f"  accepted={len(accepted)}/{len(screens)} "
            f"({100 * len(accepted) / len(screens):.1f}%)",
            flush=True,
        )
        if args.max_detectors is not None:
            accepted = accepted.head(args.max_detectors)
            print(f"  debug cap: auditing {len(accepted)} detectors", flush=True)
        observation_state = classify_observations(raw, accepted)
        transition = transition_codes(
            observation_state,
            min_valid_fraction=args.min_valid_fraction,
            min_agreement=args.min_agreement,
        )
        for local_s, sensor in enumerate(accepted["sensor"].astype(int)):
            counts = {name: int((transition[:, local_s] == code).sum())
                      for code, name in STATE_NAMES.items()}
            all_counts.append({
                "dataset": dataset,
                "sensor": sensor,
                "n_unresolved_windows": int((transition[:, local_s] < 0).sum()),
                **{f"n_{name}": value for name, value in counts.items()},
            })
        all_screens.append(screens)
        all_bins.append(bins)
        if not args.fit_only and len(accepted):
            if args.device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = args.device
            print(f"  support audit device={device}", flush=True)
            all_rows.append(analyse_support(
                dataset,
                raw,
                accepted,
                transition,
                methods=tuple(args.methods),
                ratios=tuple(args.ratios),
                seeds=tuple(args.seeds),
                device=device,
                sensor_batch=args.sensor_batch,
                query_batch=args.query_batch,
            ))

    pd.concat(all_screens, ignore_index=True).to_csv(
        OUT / f"fd_sensor_resolved_fits{args.out_suffix}.csv", index=False
    )
    pd.concat(all_bins, ignore_index=True).to_csv(
        OUT / f"fd_sensor_resolved_bins{args.out_suffix}.csv", index=False
    )
    pd.DataFrame(all_counts).to_csv(
        OUT / f"fd_sensor_resolved_state_counts{args.out_suffix}.csv", index=False
    )
    if all_rows:
        rows = pd.concat(all_rows, ignore_index=True)
        rows.to_csv(OUT / f"fd_sensor_resolved_by_detector{args.out_suffix}.csv", index=False)
        summarise_detector_rows(rows).to_csv(
            OUT / f"fd_sensor_resolved_summary{args.out_suffix}.csv", index=False
        )
    print("wrote detector-resolved FD audit artifacts", flush=True)


if __name__ == "__main__":
    main()
