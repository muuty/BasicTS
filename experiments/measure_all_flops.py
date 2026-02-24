#!/usr/bin/env python3
"""
Batch FLOPs Measurement Script

Measures FLOPs for all models across all methods and datasets,
then combines results into a single CSV file.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path and change working directory
sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


def find_all_configs():
    """Find all config files to measure."""
    base_path = Path("new_baselines_jy")
    configs = []

    # Methods and their directories
    methods = {
        "central": "central",
        "split": "split",
        "hiersplit": "hiersplit",
        "federated": "federated",
        "independent": "independent_clientwise",
    }

    # Models
    models = ["stgcn", "staeformer", "stgformer", "gruseq2seq_graphnet"]

    # Main datasets (exclude SPEED variants and special datasets)
    main_datasets = ["METR_LA", "PEMS_BAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]

    for method_name, method_dir in methods.items():
        for model in models:
            model_path = base_path / method_dir / model
            if not model_path.exists():
                continue

            for dataset in main_datasets:
                config_file = model_path / f"{dataset}.py"
                if config_file.exists():
                    configs.append({
                        "config_path": str(config_file),
                        "method": method_name,
                        "model": model,
                        "dataset": dataset.replace("_", "-"),
                    })

    return configs


def measure_single_config(config_path: str, device: torch.device, batch_size: int = 2):
    """Measure FLOPs for a single config."""
    from easytorch.config import init_cfg
    from basicts.runners.runner_zoo.flops_profiler import FLOPsProfiler

    # Import here to avoid circular imports
    from experiments.measure_flops import (
        detect_method,
        build_model_from_cfg,
        create_dummy_input,
    )

    try:
        # Load config
        cfg = init_cfg(config_path)

        # Detect method
        method = detect_method(cfg)

        # Get model and dataset names
        model_name = cfg.get("MODEL", {}).get("NAME", "unknown")
        if model_name == "unknown":
            model_cls = cfg.get("MODEL", {}).get("ARCH", None) or cfg.get("MODEL", {}).get("CLASS", None)
            if model_cls:
                model_name = model_cls.__name__
        dataset_name = cfg.get("DATASET", {}).get("NAME", "unknown")

        # Build model
        model = build_model_from_cfg(cfg, device)

        # Create dummy input
        history, future = create_dummy_input(cfg, device, batch_size)

        # Profile
        profiler = FLOPsProfiler(
            model=model,
            sample_history=history,
            sample_future=future,
            method=method,
            cfg=cfg,
        )

        with torch.no_grad():
            results = profiler.profile()

        # Add names
        results["model_name"] = model_name
        results["dataset"] = dataset_name
        results["config_path"] = config_path

        return results

    except Exception as e:
        import traceback
        print(f"    [ERROR] {e}")
        traceback.print_exc()
        return {
            "method": "error",
            "model_name": "unknown",
            "dataset": "unknown",
            "config_path": config_path,
            "error": str(e),
            "total_flops": -1,
            "total_params": -1,
            "client_flops": -1,
            "server_flops": -1,
            "client_params": -1,
            "server_params": -1,
        }


def main():
    parser = argparse.ArgumentParser(description="Batch measure FLOPs for all models")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("-o", "--output", type=str, default="flops_results/flops_all_models.csv",
                        help="Output CSV file path")
    parser.add_argument("--dry-run", action="store_true", help="Just list configs without measuring")
    args = parser.parse_args()

    # Find all configs
    configs = find_all_configs()
    print(f"Found {len(configs)} configurations to measure")

    if args.dry_run:
        for i, cfg in enumerate(configs, 1):
            print(f"  {i:3d}. [{cfg['method']:12s}] {cfg['model']:25s} - {cfg['dataset']}")
        return 0

    # Set device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Measure all configs
    all_results = []
    total = len(configs)

    for i, cfg_info in enumerate(configs, 1):
        print(f"\n[{i:3d}/{total}] Measuring: [{cfg_info['method']:12s}] {cfg_info['model']:25s} - {cfg_info['dataset']}")
        print(f"         Config: {cfg_info['config_path']}")

        start_time = time.time()
        results = measure_single_config(cfg_info['config_path'], device, args.batch_size)
        elapsed = time.time() - start_time

        # Add measurement metadata
        results["measurement_time"] = f"{elapsed:.2f}s"

        all_results.append(results)

        # Print summary
        total_flops = results.get("total_flops", -1)
        client_flops = results.get("client_flops", -1)
        server_flops = results.get("server_flops", -1)

        if total_flops > 0:
            print(f"         Total: {total_flops/1e9:.2f}G FLOPs | Client: {client_flops/1e9:.2f}G | Server: {server_flops/1e9:.2f}G ({elapsed:.1f}s)")
        else:
            print(f"         [FAILED] {results.get('error', 'Unknown error')}")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define column order
    fieldnames = [
        "method", "model_name", "dataset",
        "total_flops", "total_params",
        "client_flops", "server_flops",
        "client_params", "server_params",
        "pooler_flops", "expander_flops",
        "num_clients", "num_tokens",
        "batch_size", "input_len", "num_nodes", "num_features",
        "total_macs", "client_macs", "server_macs",
        "config_path", "measurement_time", "error",
    ]

    # Collect all unique keys
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())

    # Add any missing keys to fieldnames
    for key in sorted(all_keys):
        if key not in fieldnames:
            fieldnames.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for results in all_results:
            writer.writerow(results)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total configs: {total}")
    print(f"Successful: {sum(1 for r in all_results if r.get('total_flops', -1) > 0)}")
    print(f"Failed: {sum(1 for r in all_results if r.get('total_flops', -1) <= 0)}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
