#!/usr/bin/env python
"""Build nearest-assignment masses for the mass-mismatch experiment.

The assignment geometry is PCA-50 of standardized paired history/future
windows with L1 distance, matching the article's practical K-medoids geometry.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from easytorch.config import import_config

sys.path.append(os.path.abspath(__file__ + "/../../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from basicts.data import TimeSeriesForecastingDataset
from coreset.distance import extract_features, get_features_by_type


CONFIGS = {
    "SAN_BERNARDINO": "baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py",
    "CONTRA_COSTA": "baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py",
}
METHODS = ("k_medoids", "k_center", "graph_cut")
RATIOS = (0.1, 0.3)


def assign(features, selected, batch_size=512):
    representatives = features[selected]
    counts = np.zeros(len(selected), dtype=np.int64)
    min_distances = []
    for start in range(0, len(features), batch_size):
        distances = cdist(features[start:start + batch_size], representatives, metric="cityblock")
        nearest = distances.argmin(axis=1)
        counts += np.bincount(nearest, minlength=len(selected))
        min_distances.append(distances[np.arange(len(nearest)), nearest])
    return counts, np.concatenate(min_distances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--coreset-seed", type=int, default=42)
    args = parser.parse_args()

    for dataset_name in args.datasets:
        cfg = import_config(CONFIGS[dataset_name], verbose=False)
        dataset = TimeSeriesForecastingDataset(mode="train", **cfg["DATASET"]["PARAM"])
        inputs, targets = extract_features(dataset, cfg["MODEL"])
        raw = get_features_by_type(inputs, targets, "euclidean")
        features = PCA(n_components=50, random_state=42).fit_transform(raw).astype(np.float32)
        del raw, inputs, targets

        output_dir = Path("coreset_indices") / dataset_name / "assignment_weights"
        output_dir.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            for ratio in RATIOS:
                ratio_code = f"{ratio:.2f}".replace(".", "")
                index_path = Path("coreset_indices") / dataset_name / (
                    f"{method}_euclidean_{ratio_code}_seed{args.coreset_seed}.json"
                )
                selected = [int(value) for value in json.loads(index_path.read_text())]
                counts, min_distances = assign(features, selected)
                n, k = len(features), len(selected)
                masses = counts / n
                multipliers = k * masses
                payload = {
                    "dataset": dataset_name,
                    "method": method,
                    "ratio": ratio,
                    "coreset_seed": args.coreset_seed,
                    "geometry": "PCA-50 standardized paired history/future, L1",
                    "source_index_file": str(index_path),
                    "selected_indices": selected,
                    "cluster_counts": counts.tolist(),
                    "cluster_masses": masses.tolist(),
                    "loss_multipliers": multipliers.tolist(),
                    "quantization_cost": float(min_distances.mean()),
                    "total_variation_from_uniform": float(0.5 * np.abs(masses - 1 / k).sum()),
                    "effective_sample_size": float(1 / np.square(masses).sum()),
                }
                output_path = output_dir / f"{method}_euclidean_{ratio_code}_seed{args.coreset_seed}.json"
                output_path.write_text(json.dumps(payload, indent=2))
                print(output_path, "TV=", payload["total_variation_from_uniform"])


if __name__ == "__main__":
    main()
