#!/usr/bin/env python
"""
Offline coreset selection: pre-compute index lists and save as JSON.

Decouples selection from training so that the same indices can be reused
across multiple models without re-computing distance matrices.

Usage:
    # Generate all index files from a YAML config
    python experiments/select_coreset.py --cfg experiments/config/phase_a.yaml

    # Dry-run: show what would be generated
    python experiments/select_coreset.py --cfg experiments/config/phase_a.yaml --dry-run

    # Use specific GPU
    python experiments/select_coreset.py --cfg experiments/config/phase_a.yaml --gpus 0

Output structure:
    coreset_indices/
      {dataset_name}/
        {method}_{distance}_{ratio}_{seed}.json   # list of int indices
"""

import argparse
import inspect
import itertools
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easytorch.config import import_config
from coreset.factory import get_selection


def get_dataset_from_config(cfg):
    """Build the training dataset from a config (same logic as runner)."""
    if 'DATASET' not in cfg:
        dataset_type = cfg['TRAIN']['DATA']['DATASET']['TYPE']
        dataset_param = dict(cfg['TRAIN']['DATA']['DATASET']['PARAM'])
        if 'mode' in inspect.signature(dataset_type.__init__).parameters:
            dataset_param['mode'] = 'train'
        dataset = dataset_type(**dataset_param)
    else:
        dataset = cfg['DATASET']['TYPE'](mode='train', **cfg['DATASET']['PARAM'])
    return dataset


def get_index_filename(method: str, distance: str, ratio: float,
                       similarity: str = 'rbf', seed: int = 42) -> str:
    """Generate a deterministic filename for an index file."""
    ratio_str = f"{ratio:.2f}".replace('.', '')
    parts = [method, distance, ratio_str, f"seed{seed}"]
    if method == 'graph_cut' and similarity != 'rbf':
        parts.insert(2, similarity)
    return "_".join(parts) + ".json"


def get_dataset_name_from_config(cfg_path: str) -> str:
    """Extract dataset name from a config file for organizing output."""
    cfg = import_config(cfg_path, verbose=False)
    if 'DATASET' in cfg and 'PARAM' in cfg['DATASET']:
        return cfg['DATASET']['PARAM'].get('dataset_name', 'unknown')
    if 'TRAIN' in cfg and 'DATA' in cfg['TRAIN']:
        return cfg['TRAIN']['DATA']['DATASET']['PARAM'].get('dataset_name', 'unknown')
    return 'unknown'


def run_selection(cfg_path: str, method: str, ratio: float, distance: str,
                  similarity: str, seed: int, output_dir: Path, dry_run: bool = False) -> str:
    """Run a single coreset selection and save the result.

    Returns the output file path.
    """
    filename = get_index_filename(method, distance, ratio, similarity, seed)
    output_path = output_dir / filename

    if output_path.exists():
        print(f"  [SKIP] {filename} already exists")
        return str(output_path)

    if dry_run:
        print(f"  [DRY] {filename}")
        return str(output_path)

    print(f"  [RUN]  {filename} ...", end=" ", flush=True)
    start = time.time()

    cfg = import_config(cfg_path, verbose=False)
    dataset = get_dataset_from_config(cfg)
    model_config = cfg['MODEL']

    selection = get_selection(
        type=method,
        selection_ratio=ratio,
        dataset=dataset,
        model_config=model_config,
        distance_type=distance,
        similarity_type=similarity,
        seed=seed,
    )
    indices = selection.select_indices()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(indices, f)

    elapsed = time.time() - start
    print(f"done ({elapsed:.1f}s, {len(indices)} indices)")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Offline coreset selection")
    parser.add_argument("--cfg", type=str, required=True, help="YAML config file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="coreset_indices",
                        help="Base output directory (default: coreset_indices)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    with open(args.cfg) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.cfg}")
    print(f"Output: {args.output_dir}/\n")

    # Collect unique (config, method, ratio, distance, similarity, seed) combos
    # across all runs. Different configs (models) sharing the same dataset
    # will produce the same indices, so we only need one config per dataset.
    seen = {}  # dataset_name -> first config path
    all_selections = []

    for exp in config['runs']:
        params = exp.get('params', {})

        methods = params.get('CORESET.SELECTION_STRATEGY', ['k_medoids'])
        ratios = params.get('CORESET.SELECTION_RATIO', [0.5])
        distances = params.get('CORESET.DISTANCE_TYPE', ['euclidean'])
        similarities = params.get('CORESET.SIMILARITY_TYPE', ['rbf'])
        seeds = params.get('CORESET.SEED', [42])

        for cfg_path in exp['configs']:
            dataset_name = get_dataset_name_from_config(cfg_path)
            if dataset_name not in seen:
                seen[dataset_name] = cfg_path

        for method, ratio, distance, similarity, seed in itertools.product(
                methods, ratios, distances, similarities, seeds):
            # Skip non-distance methods (selection is trivial)
            if method in ('random', 'recent', 'stride') and distance != distances[0]:
                continue
            for dataset_name in seen:
                key = (dataset_name, method, ratio, distance, similarity, seed)
                if key not in {s[0] for s in all_selections}:
                    all_selections.append(
                        (key, seen[dataset_name], dataset_name,
                         method, ratio, distance, similarity, seed)
                    )

    print(f"Total unique selections: {len(all_selections)}\n")

    for _, cfg_path, dataset_name, method, ratio, distance, similarity, seed in all_selections:
        # Clean dataset name for path (e.g., "xtraffic/SAN_BERNARDINO" -> "SAN_BERNARDINO")
        clean_name = dataset_name.replace('xtraffic/', '').replace('/', '_')
        output_dir = Path(args.output_dir) / clean_name
        print(f"[{clean_name}] {method} ratio={ratio} dist={distance} seed={seed}")
        run_selection(cfg_path, method, ratio, distance, similarity, seed,
                      output_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
