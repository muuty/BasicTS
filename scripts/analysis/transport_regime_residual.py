#!/usr/bin/env python3
"""Audit the input-only future-trajectory residual by traffic regime.

This is a deterministic geometry diagnostic for Corollary 1, not a neural
network experiment or a certified risk bound.  Histories are represented by
PCA-12 projections of network mean/dispersion trajectories.  Future distance
is mean absolute difference over the complete 12-step, all-sensor, three-
channel target tensor.  Rather than selecting one favourable continuity slope,
the script audits K at 0 and at the 50th, 75th, 90th, and 95th percentiles of
d_Y/d_X among nearest non-self training histories.

Operational regimes are defined without external incident labels:
  * weekday peak versus other periods;
  * congested future windows (bottom quintile of network mean speed);
  * abrupt speed-drop windows (bottom decile of future-minus-history speed).

Outputs:
  experiments/result/analysis/transport_regime_residual_by_config.csv
  experiments/result/analysis/transport_regime_residual_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
DATASETS = ["SAN_BERNARDINO", "CONTRA_COSTA"]
METHODS = ["k_medoids", "random", "stride", "recent", "k_center", "graph_cut"]
SEEDS = [42, 123, 456]
INPUT_LEN = 12
OUTPUT_LEN = 12
DATA_RANGE_END = 24192
VALID_RATIO = TEST_RATIO = 0.2
TRAIN_TIMESTEPS = (
    DATA_RANGE_END
    - int(DATA_RANGE_END * VALID_RATIO)
    - int(DATA_RANGE_END * TEST_RATIO)
)
N_WINDOWS = TRAIN_TIMESTEPS - INPUT_LEN - OUTPUT_LEN + 1


def load_traffic(dataset: str) -> np.ndarray:
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    shape = tuple(desc["shape"])
    data = np.memmap(
        REPO / "datasets" / dataset / "data.dat",
        dtype=np.float32,
        mode="r",
        shape=shape,
    )
    return np.asarray(data[:TRAIN_TIMESTEPS, :, :3], dtype=np.float32)


def window_summary(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 72-D history and future network-level trajectory summaries."""
    step = np.concatenate([z.mean(axis=1), z.std(axis=1)], axis=1)
    starts = np.arange(N_WINDOWS)[:, None]
    offsets = np.arange(INPUT_LEN)[None, :]
    history = step[starts + offsets].reshape(N_WINDOWS, -1)
    future = step[starts + INPUT_LEN + offsets].reshape(N_WINDOWS, -1)
    return history.astype(np.float32), future.astype(np.float32)


def future_l1(z: np.ndarray, left: np.ndarray, right: np.ndarray,
              batch_size: int = 192) -> np.ndarray:
    """Mean L1 target distance for paired window-start arrays."""
    out = np.empty(len(left), dtype=np.float32)
    offsets = np.arange(OUTPUT_LEN)[None, :]
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        li = left[start:stop, None] + INPUT_LEN + offsets
        ri = right[start:stop, None] + INPUT_LEN + offsets
        out[start:stop] = np.abs(z[li] - z[ri]).mean(axis=(1, 2, 3))
    return out


def build_regimes(raw: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    starts = np.arange(N_WINDOWS)
    future_start = starts + INPUT_LEN
    day = (future_start // 288) % 7  # data range starts Sunday; 0=Sun
    hour = (future_start % 288) // 12
    weekday = (day >= 1) & (day <= 5)
    peak = weekday & (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19)))

    speed = raw[:, :, 2].mean(axis=1)
    offsets = np.arange(OUTPUT_LEN)[None, :]
    idx = starts[:, None]
    history_speed = speed[idx + np.arange(INPUT_LEN)[None, :]].mean(axis=1)
    future_speed = speed[idx + INPUT_LEN + offsets].mean(axis=1)
    speed_delta = future_speed - history_speed
    congestion_threshold = float(np.quantile(future_speed, 0.20))
    drop_threshold = float(np.quantile(speed_delta, 0.10))
    congested = future_speed <= congestion_threshold
    abrupt_drop = speed_delta <= drop_threshold

    regimes = {
        "all": np.ones(N_WINDOWS, dtype=bool),
        "peak": peak,
        "off_peak": ~peak,
        "weekday": weekday,
        "weekend": ~weekday,
        "congested_q20": congested,
        "noncongested": ~congested,
        "abrupt_speed_drop_q10": abrupt_drop,
        "non_drop": ~abrupt_drop,
    }
    thresholds = {
        "congestion_speed_q20": congestion_threshold,
        "speed_delta_q10": drop_threshold,
    }
    return regimes, thresholds


def load_indices(dataset: str, method: str, ratio: float, seed: int) -> np.ndarray:
    ratio_pct = int(round(100 * ratio))
    path = (
        REPO / "coreset_indices" / dataset
        / f"{method}_euclidean_{ratio_pct:03d}_seed{seed}.json"
    )
    selected = np.asarray(json.loads(path.read_text()), dtype=np.int64)
    if selected.min() < 0 or selected.max() >= N_WINDOWS:
        raise ValueError(f"Out-of-range index in {path}")
    return selected


def analyse_dataset(dataset: str, ratios: list[float]) -> list[dict]:
    print(f"[{dataset}] loading and standardising traffic tensor", flush=True)
    raw = load_traffic(dataset)
    mean = raw.mean(axis=(0, 1), keepdims=True)
    std = raw.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    z = (raw - mean) / std

    history, _future = window_summary(z)
    h_mean = history.mean(axis=0, keepdims=True)
    h_std = history.std(axis=0, keepdims=True)
    h_std[h_std == 0] = 1.0
    history = (history - h_mean) / h_std
    pca = PCA(n_components=12, random_state=42)
    history_pca = pca.fit_transform(history).astype(np.float32)
    print(
        f"  history PCA-12 variance={pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )

    # Calibrate a declared K curve on nearest non-self histories.  The
    # diagnostic is deterministic at every point on the curve.
    local_nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    local_nn.fit(history_pca)
    local_dist, local_idx = local_nn.kneighbors(history_pca, return_distance=True)
    dx_local = local_dist[:, 1]
    dy_local = future_l1(z, np.arange(N_WINDOWS), local_idx[:, 1])
    valid = dx_local > 1e-8
    local_slope = dy_local[valid] / dx_local[valid]
    K_values = {
        "K0": 0.0,
        "K50": float(np.quantile(local_slope, 0.50)),
        "K75": float(np.quantile(local_slope, 0.75)),
        "K90": float(np.quantile(local_slope, 0.90)),
        "K95": float(np.quantile(local_slope, 0.95)),
    }
    regimes, thresholds = build_regimes(raw)
    print(
        f"  K curve={K_values}, congestion_q20={thresholds['congestion_speed_q20']:.3f}, "
        f"drop_q10={thresholds['speed_delta_q10']:.3f}",
        flush=True,
    )

    rows: list[dict] = []
    all_idx = np.arange(N_WINDOWS)
    for ratio in ratios:
        for method in METHODS:
            for seed in SEEDS:
                selected = load_indices(dataset, method, ratio, seed)
                nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
                nn.fit(history_pca[selected])
                dx, assignment = nn.kneighbors(history_pca, return_distance=True)
                dx = dx[:, 0]
                representative = selected[assignment[:, 0]]
                dy = future_l1(z, all_idx, representative)
                for K_label, K in K_values.items():
                    residual = np.maximum(dy - K * dx, 0.0)
                    data_term = K * dx + residual
                    for regime, mask in regimes.items():
                        rows.append({
                            "dataset": dataset,
                            "method": method,
                            "ratio": ratio,
                            "selection_seed": seed,
                            "regime": regime,
                            "n_windows": int(mask.sum()),
                            "K_label": K_label,
                            "K": K,
                            "history_pca_variance": float(
                                pca.explained_variance_ratio_.sum()
                            ),
                            **thresholds,
                            "J_X": float(dx[mask].mean()),
                            "future_gap": float(dy[mask].mean()),
                            "residual": float(residual[mask].mean()),
                            "data_term_KJ_plus_residual": float(data_term[mask].mean()),
                            "residual_positive_rate": float((residual[mask] > 0).mean()),
                        })
                print(
                    f"  r={ratio:.1f} {method:10s} seed={seed}: "
                    f"Jx={dx.mean():.4f} future_gap={dy.mean():.4f}",
                    flush=True,
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.1, 0.3],
        help="Selection ratios to analyse (default: 0.1 0.3)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        rows.extend(analyse_dataset(dataset, args.ratios))
    by_config = pd.DataFrame(rows)
    config_out = OUT / "transport_regime_residual_by_config.csv"
    by_config.to_csv(config_out, index=False)

    summary = (
        by_config.groupby(
            ["method", "ratio", "regime", "K_label"], as_index=False
        )
        .agg(
            n_dataset_seeds=("selection_seed", "size"),
            K=("K", "mean"),
            J_X=("J_X", "mean"),
            future_gap=("future_gap", "mean"),
            residual=("residual", "mean"),
            data_term_KJ_plus_residual=("data_term_KJ_plus_residual", "mean"),
            residual_positive_rate=("residual_positive_rate", "mean"),
        )
    )
    summary_out = OUT / "transport_regime_residual_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"wrote {config_out} ({len(by_config)} rows)")
    print(f"wrote {summary_out} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
