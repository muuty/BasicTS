#!/usr/bin/env python3
"""Stratify network-wide joint quantization by detector traffic state.

This preliminary analysis uses the same PCA-50, paired history/future, L1
geometry as the assignment-mass experiment.  Each training window has one
network-wide nearest-representative distance.  That distance is then attached
to every resolved detector-state atom in the window; assignments are not
recomputed separately per detector.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(__file__ + "/../../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from easytorch.config import import_config
from basicts.data import TimeSeriesForecastingDataset
from coreset.distance import extract_features, get_features_by_type
from scripts.analysis.fd_sensor_resolved import (
    STATE_NAMES,
    classify_observations,
    load_raw,
    transition_codes,
)


REPO = Path.cwd()
OUT = REPO / "experiments" / "result" / "analysis"
CONFIGS = {
    "SAN_BERNARDINO": "baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py",
    "CONTRA_COSTA": "baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py",
}
METHODS = ("k_medoids", "k_center", "graph_cut")
RATIOS = (0.1, 0.3)


def nearest_distance(features: np.ndarray, selected: list[int], batch_size: int) -> np.ndarray:
    support = features[selected]
    values = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        distances = cdist(features[start:start + batch_size], support, metric="cityblock")
        values[start:start + len(distances)] = distances.min(axis=1)
    return values


def bootstrap_ratio(values: pd.DataFrame, draws: int = 5000) -> tuple[float, float, float]:
    pivot = values.pivot(index="sensor", columns="state", values="quantization").dropna()
    ratio = pivot.drop(columns="free_to_free").div(pivot["free_to_free"], axis=0)
    rng = np.random.default_rng(42)
    output = {}
    for state in ratio:
        array = ratio[state].to_numpy()
        estimates = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
        output[state] = (
            float(array.mean()),
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument("--coreset-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    fits = pd.read_csv(OUT / "fd_sensor_resolved_fits.csv")
    detector_rows, summary_rows = [], []
    for dataset_name in args.datasets:
        print(f"[{dataset_name}] features and states", flush=True)
        cfg = import_config(CONFIGS[dataset_name], verbose=False)
        dataset = TimeSeriesForecastingDataset(mode="train", **cfg["DATASET"]["PARAM"])
        inputs, targets = extract_features(dataset, cfg["MODEL"])
        raw_features = get_features_by_type(inputs, targets, "euclidean")
        features = PCA(n_components=50, random_state=42).fit_transform(raw_features).astype(np.float32)
        del inputs, targets, raw_features

        accepted = fits[(fits["dataset"] == dataset_name) & fits["identifiable"]].sort_values("sensor")
        raw = load_raw(dataset_name)
        observation = classify_observations(raw, accepted)
        transitions = transition_codes(observation, min_valid_fraction=2 / 3, min_agreement=0.75)
        sensors = accepted["sensor"].astype(int).to_numpy()

        for method in args.methods:
            for ratio in args.ratios:
                ratio_code = f"{ratio:.2f}".replace(".", "")
                index_path = REPO / "coreset_indices" / dataset_name / (
                    f"{method}_euclidean_{ratio_code}_seed{args.coreset_seed}.json"
                )
                selected = [int(value) for value in json.loads(index_path.read_text())]
                qdist = nearest_distance(features, selected, args.batch_size)
                is_selected = np.zeros(len(features), dtype=bool)
                is_selected[selected] = True
                config_rows = []
                for local_sensor, sensor in enumerate(sensors):
                    for code, state in STATE_NAMES.items():
                        mask = transitions[:, local_sensor] == code
                        if not mask.any():
                            continue
                        row = {
                            "dataset": dataset_name,
                            "sensor": int(sensor),
                            "method": method,
                            "ratio": ratio,
                            "coreset_seed": args.coreset_seed,
                            "state": state,
                            "n_atoms": int(mask.sum()),
                            "quantization": float(qdist[mask].mean()),
                            "selected_fraction": float(is_selected[mask].mean()),
                        }
                        detector_rows.append(row)
                        config_rows.append(row)

                frame = pd.DataFrame(config_rows)
                ratios = bootstrap_ratio(frame)
                full_counts = frame.groupby("state")["n_atoms"].sum()
                selected_counts = {}
                for state in STATE_NAMES.values():
                    atom_mask = transitions == list(STATE_NAMES.keys())[list(STATE_NAMES.values()).index(state)]
                    selected_counts[state] = int(atom_mask[is_selected].sum())
                selected_counts = pd.Series(selected_counts)
                full_share = full_counts / full_counts.sum()
                selected_share = selected_counts / selected_counts.sum()
                for state, group in frame.groupby("state"):
                    ratio_est, ratio_low, ratio_high = (1.0, 1.0, 1.0)
                    if state != "free_to_free":
                        ratio_est, ratio_low, ratio_high = ratios[state]
                    summary_rows.append({
                        "dataset": dataset_name,
                        "method": method,
                        "ratio": ratio,
                        "coreset_seed": args.coreset_seed,
                        "state": state,
                        "n_detectors": group["sensor"].nunique(),
                        "n_atoms": int(group["n_atoms"].sum()),
                        "quantization_detector_mean": float(group["quantization"].mean()),
                        "quantization_ratio_to_free": ratio_est,
                        "ratio_ci_low": ratio_low,
                        "ratio_ci_high": ratio_high,
                        "selected_fraction_detector_mean": float(group["selected_fraction"].mean()),
                        "selected_state_share": float(selected_share[state]),
                        "full_state_share": float(full_share[state]),
                        "selection_state_lift": float(selected_share[state] / full_share[state]),
                    })
                print(f"  {method:10s} r={ratio:.1f} global_q={qdist.mean():.3f}", flush=True)

    detector_output = pd.DataFrame(detector_rows)
    summary_output = pd.DataFrame(summary_rows)
    detector_output.to_csv(OUT / "state_stratified_quantization_by_detector.csv", index=False)
    summary_output.to_csv(OUT / "state_stratified_quantization_summary.csv", index=False)
    print(summary_output.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
