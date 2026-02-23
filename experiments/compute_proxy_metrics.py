#!/usr/bin/env python
"""
Compute all 6 proxy metrics for pre-computed coreset index files.

Metrics:
  1. OT Cost (vanilla Sinkhorn)
  2. Sinkhorn Divergence (debiased)
  3. Facility Location Objective
  4. Intra-coreset Redundancy
  5. Information Gain (FL - λ·Redundancy)
  6. Temporal Diversity (H_tod, H_dow)

Usage:
    python experiments/compute_proxy_metrics.py \
        --cfg experiments/config/phase_a_distance_screening.yaml --gpus 0

    # Only recompute missing metrics (skip already-computed)
    python experiments/compute_proxy_metrics.py \
        --cfg experiments/config/phase_a_distance_screening.yaml --gpus 0 --skip-existing
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easytorch.config import import_config
from experiments.select_coreset import (
    get_dataset_from_config,
    get_dataset_name_from_config,
)


def compute_metrics_for_dataset(
    cfg_path: str,
    dataset_name: str,
    index_dir: Path,
    pca_dim: int = 50,
    epsilon: float = 0.1,
    subsample_full: int = 5000,
    lam: float = 1.0,
    skip_existing: bool = False,
) -> dict:
    """Compute all 6 proxy metrics for all index files in a dataset directory."""
    from coreset.distance import (
        extract_features, get_features_by_type, compute_distance_matrix
    )
    from coreset.ot_distance import OTDistanceCalculator, _sinkhorn_cost
    from coreset.proxy_metrics import (
        build_similarity_rbf, compute_combinatorial_metrics,
        compute_temporal_diversity,
    )

    # Load existing metrics if any
    metrics_path = index_dir / "proxy_metrics.json"
    existing_metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            existing_metrics = json.load(f)

    # Find all index files
    index_files = sorted(index_dir.glob("*.json"))
    index_files = [f for f in index_files if f.name != "proxy_metrics.json"]

    if not index_files:
        print(f"  No index files found in {index_dir}")
        return existing_metrics

    # Load dataset and extract features once
    print(f"  Loading dataset from {cfg_path}...")
    cfg = import_config(cfg_path, verbose=False)
    dataset = get_dataset_from_config(cfg)
    model_config = cfg['MODEL']
    dataset_size = len(dataset)

    print(f"  Extracting features ({dataset_size} samples)...")
    t0 = time.time()
    inputs, targets = extract_features(dataset, model_config)
    print(f"  Features extracted in {time.time() - t0:.1f}s")

    # Identify distance types needed
    KNOWN_DISTANCES = {'euclidean', 'temporal', 'spatial', 'combined'}
    distance_types = set()
    for f in index_files:
        for part in f.stem.split('_'):
            if part in KNOWN_DISTANCES:
                distance_types.add(part)
                break

    # Pre-compute features per distance type
    features_cache = {}
    for dt in distance_types:
        print(f"  Pre-computing {dt} features...", end=" ", flush=True)
        features_cache[dt] = get_features_by_type(inputs, targets, dt)
        print(f"dim={features_cache[dt].shape[1]}")

    # Pre-compute OT calculators (PCA fitted on full features)
    ot_calculators = {}
    for dt in distance_types:
        calc = OTDistanceCalculator(pca_dim=pca_dim, epsilon=epsilon)
        calc.fit_pca(features_cache[dt])
        ot_calculators[dt] = calc

    # Pre-compute distance matrices + similarity matrices per distance type
    # (shared across all index files of the same distance type)
    print(f"  Building similarity matrices...")
    sim_cache = {}
    for dt in distance_types:
        print(f"    {dt}...", end=" ", flush=True)
        t0 = time.time()
        dist_mat = compute_distance_matrix(inputs, targets, dt)
        sim_mat = build_similarity_rbf(dist_mat)
        sim_cache[dt] = sim_mat
        print(f"done ({time.time() - t0:.1f}s)")

    # Compute metrics for each index file
    metrics = dict(existing_metrics)
    total = len(index_files)
    for i, index_file in enumerate(index_files, 1):
        name = index_file.name

        if skip_existing and name in existing_metrics:
            print(f"  [{i}/{total}] SKIP {name}")
            continue

        # Parse distance type from filename (handle multi-word methods like k_center)
        distance_type = next(
            (p for p in index_file.stem.split('_') if p in KNOWN_DISTANCES),
            'euclidean'
        )

        with open(index_file, 'r') as f:
            indices = json.load(f)

        print(f"  [{i}/{total}] {name} ({len(indices)} indices)...", end=" ", flush=True)
        t0 = time.time()

        # --- OT metrics (#1, #2) ---
        calc = ot_calculators[distance_type]
        features = features_cache[distance_type]
        coreset_features = features[indices]

        X, Y = calc._prepare(coreset_features, features, subsample_full)
        ot_cost = _sinkhorn_cost(X, Y, calc.epsilon, calc.max_iter)
        ot_pp = _sinkhorn_cost(X, X, calc.epsilon, calc.max_iter)
        ot_qq = _sinkhorn_cost(Y, Y, calc.epsilon, calc.max_iter)
        sinkhorn_div = max(0.0, ot_cost - 0.5 * ot_pp - 0.5 * ot_qq)

        # --- Combinatorial metrics (#3, #4, #5) ---
        sim = sim_cache[distance_type]
        comb = compute_combinatorial_metrics(sim, indices, lam=lam)

        # --- Temporal diversity (#6) ---
        temp_div = compute_temporal_diversity(indices, dataset_size)

        elapsed = time.time() - t0
        ratio = len(indices) / dataset_size

        metrics[name] = {
            # OT-based (distributional)
            "ot_cost": round(ot_cost, 6),
            "sinkhorn_divergence": round(sinkhorn_div, 6),
            # Combinatorial
            "fl_objective": round(comb['fl_objective'], 4),
            "redundancy": round(comb['redundancy'], 4),
            "information_gain": round(comb['information_gain'], 4),
            # Temporal
            "h_tod": temp_div['h_tod'],
            "h_dow": temp_div['h_dow'],
            # Meta
            "num_indices": len(indices),
            "total_samples": dataset_size,
            "ratio": round(ratio, 4),
            "distance_type": distance_type,
            "compute_time_s": round(elapsed, 2),
        }
        print(f"SD={sinkhorn_div:.6f} FL={comb['fl_objective']:.1f} "
              f"H_tod={temp_div['h_tod']:.3f} ({elapsed:.1f}s)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute proxy metrics for coreset selections")
    parser.add_argument("--cfg", type=str, required=True, help="YAML config file")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--index-dir", type=str, default="coreset_indices")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--subsample", type=int, default=5000)
    parser.add_argument("--lam", type=float, default=1.0, help="Lambda for Information Gain")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip metrics already in proxy_metrics.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    with open(args.cfg) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.cfg}")
    print(f"Index dir: {args.index_dir}/\n")

    # Find unique datasets and their config paths
    seen = {}
    for exp in config['runs']:
        for cfg_path in exp['configs']:
            dataset_name = get_dataset_name_from_config(cfg_path)
            if dataset_name not in seen:
                seen[dataset_name] = cfg_path

    for dataset_name, cfg_path in seen.items():
        clean_name = dataset_name.replace('xtraffic/', '').replace('/', '_')
        index_dir = Path(args.index_dir) / clean_name
        print(f"\n=== {clean_name} ===")

        if not index_dir.exists():
            print(f"  Directory {index_dir} not found, skipping")
            continue

        metrics = compute_metrics_for_dataset(
            cfg_path=cfg_path,
            dataset_name=dataset_name,
            index_dir=index_dir,
            pca_dim=args.pca_dim,
            epsilon=args.epsilon,
            subsample_full=args.subsample,
            lam=args.lam,
            skip_existing=args.skip_existing,
        )

        # Save metrics
        metrics_path = index_dir / "proxy_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved {len(metrics)} metrics to {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
