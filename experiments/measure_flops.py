#!/usr/bin/env python3
"""
FLOPs Measurement Script

Usage:
    # Central model
    python experiments/measure_flops.py -c baselines/STAEformer/METR-LA.py
    
    # Split learning
    python experiments/measure_flops.py -c new_baselines_jy/split/staeformer/METR_LA.py --method split
    
    # HierSplit
    python experiments/measure_flops.py -c new_baselines_jy/hiersplit/staeformer/METR_LA.py --method hiersplit
    
    # With custom output directory
    python experiments/measure_flops.py -c config.py --output-dir results/flops
    
    # Specify GPU
    python experiments/measure_flops.py -c config.py -g 0
"""

import argparse
import os
import sys

# Add project root to path and change working directory
sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from easytorch.config import init_cfg

from basicts.runners.runner_zoo.flops_profiler import (
    FLOPsProfiler,
    save_flops_csv,
    print_flops_summary,
)


def detect_method(cfg: dict) -> str:
    """
    Detect learning method from config.
    """
    runner_cls = cfg.get("RUNNER", None)
    if runner_cls is None:
        return "central"
    
    runner_name = runner_cls.__name__.lower()
    
    if "hiersplit" in runner_name or "hierarchicalsplit" in runner_name:
        return "hiersplit"
    elif "split" in runner_name:
        return "split"
    elif "federated" in runner_name:
        return "federated"
    elif "independent" in runner_name:
        return "independent"
    else:
        return "central"


def build_model_from_cfg(cfg: dict, device: torch.device) -> torch.nn.Module:
    """
    Build model from config without initializing full runner.
    """
    model_cls = cfg.get("MODEL", {}).get("ARCH", None) or cfg.get("MODEL", {}).get("CLASS", None)
    if model_cls is None:
        raise ValueError("Config must have MODEL.ARCH or MODEL.CLASS")
    
    model_params = dict(cfg.get("MODEL", {}).get("PARAM", {}))
    
    # Handle special cases for split/hiersplit models
    runner_cls = cfg.get("RUNNER", None)
    if runner_cls is not None:
        runner_name = runner_cls.__name__.lower()
        
        if "split" in runner_name or "hiersplit" in runner_name or "independent" in runner_name or "federated" in runner_name:
            # These models need partition info
            return _build_partitioned_model(cfg, device)
    
    # Standard model
    model = model_cls(**model_params)
    model = model.to(device)
    model.eval()
    return model


def _build_partitioned_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """
    Build partitioned model (split/hiersplit/independent/federated).
    """
    from basicts.runners.node_partition import setup_split_learning_nodes
    from basicts.utils.serialization import load_adj
    
    # Get IL config
    il = cfg.get("IL", {})
    num_clients = int(il.get("NUM_CLIENTS", 10))
    grouping_method = str(il.get("GROUPING_METHOD", "random")).lower()
    partition_seed = il.get("PARTITION_SEED", 0)
    
    # Get dataset config
    ds = cfg.get("DATASET", {})
    model_params = dict(cfg.get("MODEL", {}).get("PARAM", {}))
    total_nodes = ds.get("NUM_NODES", None) or model_params.get("num_nodes", 207)
    adj_matrix = ds.get("ADJ_MX", None)
    dataset_name = ds.get("NAME", None)
    
    # Try to load adj_matrix from file if not provided in config
    if adj_matrix is None and dataset_name is not None:
        adj_path = f"datasets/{dataset_name}/adj_mx.pkl"
        try:
            import os
            if os.path.exists(adj_path):
                adj_mx, _ = load_adj(adj_path, "normlap")
                adj_matrix = torch.Tensor(adj_mx[0])
                print(f"Loaded adj_matrix from {adj_path}, shape: {adj_matrix.shape}")
        except Exception as e:
            print(f"[WARNING] Failed to load adj_matrix from {adj_path}: {e}")
    
    # Setup partition
    client_nodes_list, subgraph_adj_list, _ = setup_split_learning_nodes(
        num_nodes=total_nodes,
        num_clients=num_clients,
        grouping_method=grouping_method,
        adj_matrix=adj_matrix,
        dataset_name=dataset_name,
        random_seed=partition_seed,
    )
    
    # Get full adj
    full_adj = adj_matrix
    
    # Detect model type
    runner_cls = cfg.get("RUNNER", None)
    runner_name = runner_cls.__name__.lower() if runner_cls else ""
    
    base_cls = cfg.get("MODEL", {}).get("ARCH", None) or cfg.get("MODEL", {}).get("CLASS", None)
    
    if "hiersplit" in runner_name or "hierarchicalsplit" in runner_name:
        # HierSplit model
        from basicts.runners.runner_zoo.jy_hiersplit_learning_runner import HierSplitModel
        
        hs = cfg.get("HIERSPLIT", {})
        pooling = str(hs.get("POOLING", "attention")).lower()
        expansion = str(hs.get("EXPANSION", pooling)).lower()
        num_tokens = int(hs.get("NUM_TOKENS", 4))
        token_heads = int(hs.get("TOKEN_HEADS", 4))
        server_heads = int(hs.get("SERVER_HEADS", 4))
        server_layers = int(hs.get("SERVER_LAYERS", 1))
        server_ff_dim = int(hs.get("SERVER_FF_DIM", 256))
        dropout = float(hs.get("DROPOUT", 0.1))
        
        model = HierSplitModel(
            base_model_cls=base_cls,
            base_model_params=model_params,
            client_nodes_list=client_nodes_list,
            subgraph_adj_list=subgraph_adj_list,
            total_nodes=total_nodes,
            pooling=pooling,
            expansion=expansion,
            num_tokens=num_tokens,
            token_heads=token_heads,
            server_heads=server_heads,
            server_layers=server_layers,
            server_ff_dim=server_ff_dim,
            dropout=dropout,
        )
    elif "split" in runner_name:
        # Split model
        from basicts.runners.runner_zoo.jy_split_learning_runner import SplitLearningModel
        
        kind = cfg.get("SPLIT", {}).get("KIND", None)
        if kind is None:
            name = base_cls.__name__.lower()
            if "staeformer" in name:
                kind = "staeformer"
            elif "stgformer" in name:
                kind = "stgformer"
            elif "stgcn" in name:
                kind = "stgcn"
            else:
                kind = "gruseq2seq_graphnet"
        
        model = SplitLearningModel(
            base_model_cls=base_cls,
            base_model_params=model_params,
            client_nodes_list=client_nodes_list,
            total_nodes=total_nodes,
            kind=kind,
            full_adj=full_adj,
        )
    else:
        # Independent/Federated model
        from basicts.runners.runner_zoo.jy_independent_ensemble import IndependentClientEnsemble
        
        output_dim = model_params.get("output_dim", 1)
        
        model = IndependentClientEnsemble(
            base_model_class=base_cls,
            base_model_params=model_params,
            client_nodes_list=client_nodes_list,
            subgraph_adj_list=subgraph_adj_list,
            full_adj=full_adj,
            output_dim=output_dim,
        )
    
    model = model.to(device)
    model.eval()
    return model


def get_batch_size_from_config(cfg) -> int:
    """
    Read batch size from config. Tries TRAIN, then TEST, then VAL DATA.BATCH_SIZE.
    Returns 2 if none found.
    """
    for phase in ("TRAIN", "TEST", "VAL"):
        try:
            phase_cfg = getattr(cfg, phase, None)
            if phase_cfg is None:
                continue
            data = getattr(phase_cfg, "DATA", None)
            if data is None:
                continue
            bs = getattr(data, "BATCH_SIZE", None)
            if bs is not None:
                return int(bs)
        except Exception:
            continue
    return 2


def create_input_from_dataset(cfg, device: torch.device, batch_size: int) -> tuple:
    """
    Create input tensors from actual dataset.
    This ensures correct data format including proper embedding indices.
    """
    from torch.utils.data import DataLoader
    from basicts.data import TimeSeriesForecastingDataset
    
    # Get dataset parameters from cfg (easytorch Config object)
    ds_param = cfg.DATASET.PARAM
    dataset_name = ds_param.dataset_name
    train_val_test_ratio = list(ds_param.train_val_test_ratio)
    input_len = ds_param.input_len
    output_len = ds_param.output_len
    overlap = getattr(ds_param, 'overlap', True)
    
    # Build dataset with train mode
    dataset = TimeSeriesForecastingDataset(
        dataset_name=dataset_name,
        train_val_test_ratio=train_val_test_ratio,
        mode="train",
        input_len=input_len,
        output_len=output_len,
        overlap=overlap,
    )
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get one batch - dataset returns dict with 'inputs' and 'target' keys
    batch = next(iter(loader))
    
    # batch is dict: {'inputs': tensor, 'target': tensor}
    # shapes: [B, T, N, C]
    inputs = batch['inputs']
    targets = batch['target']
    
    # Select forward features
    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    
    # Apply feature selection
    if forward_features:
        history = inputs[:, :, :, forward_features].float().to(device)
        future = targets[:, :, :, forward_features].float().to(device)
    else:
        history = inputs.float().to(device)
        future = targets.float().to(device)
    
    return history, future


def create_dummy_input(cfg: dict, device: torch.device, batch_size: int) -> tuple:
    """
    Create input tensors - tries real dataset first, falls back to synthetic.
    """
    # Try to load from actual dataset
    try:
        history, future = create_input_from_dataset(cfg, device, batch_size)
        print(f"Loaded real data from dataset")
        return history, future
    except Exception as e:
        print(f"[WARNING] Could not load real dataset: {e}")
        print("Falling back to synthetic data...")
    
    # Fallback: create synthetic data
    model_params = cfg.get("MODEL", {}).get("PARAM", {})
    ds = cfg.get("DATASET", {})
    
    ds_param = ds.get("PARAM", {})
    input_len = ds_param.get("input_len", 12)
    output_len = ds_param.get("output_len", 12)
    
    num_nodes = ds.get("NUM_NODES", None) or model_params.get("num_nodes", 207)
    
    forward_features = cfg.get("MODEL", {}).get("FORWARD_FEATURES", [0])
    num_features = len(forward_features) if forward_features else 1
    
    # Create tensors with valid embedding indices
    history = torch.zeros(batch_size, input_len, num_nodes, num_features, device=device)
    future = torch.zeros(batch_size, output_len, num_nodes, num_features, device=device)
    
    # Feature 0: traffic data (random)
    history[:, :, :, 0] = torch.randn(batch_size, input_len, num_nodes, device=device)
    future[:, :, :, 0] = torch.randn(batch_size, output_len, num_nodes, device=device)
    
    # Feature 1: time-of-day index (normalized 0-1)
    if num_features > 1:
        tod_indices = torch.randint(0, 288, (batch_size, input_len, num_nodes), device=device)
        history[:, :, :, 1] = tod_indices.float() / 288.0
        tod_indices_f = torch.randint(0, 288, (batch_size, output_len, num_nodes), device=device)
        future[:, :, :, 1] = tod_indices_f.float() / 288.0
    
    # Feature 2: day-of-week index (normalized 0-1)
    if num_features > 2:
        dow_indices = torch.randint(0, 7, (batch_size, input_len, num_nodes), device=device)
        history[:, :, :, 2] = dow_indices.float() / 7.0
        dow_indices_f = torch.randint(0, 7, (batch_size, output_len, num_nodes), device=device)
        future[:, :, :, 2] = dow_indices_f.float() / 7.0
    
    return history, future


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs for time series forecasting models")
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument("--method", type=str, default=None, 
                        help="Method type: central, independent, split, hiersplit, federated (auto-detected if not specified)")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                        help="Batch size for profiling (default: from config TRAIN/TEST/VAL.DATA.BATCH_SIZE, else 2)")
    parser.add_argument("--output-dir", type=str, default="flops_results", help="Output directory for CSV")
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load config
    print(f"Loading config: {args.config}")
    cfg = init_cfg(args.config)

    # Batch size: CLI override > config (TRAIN/TEST/VAL.DATA.BATCH_SIZE) > 2
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = get_batch_size_from_config(cfg)
        print(f"Using batch size from config: {batch_size}")
    else:
        print(f"Using batch size from CLI (-b): {batch_size}")
    
    # Detect method if not specified
    method = args.method or detect_method(cfg)
    print(f"Method: {method}")
    
    # Get model and dataset names
    model_name = cfg.get("MODEL", {}).get("NAME", "unknown")
    if model_name == "unknown":
        model_cls = cfg.get("MODEL", {}).get("ARCH", None) or cfg.get("MODEL", {}).get("CLASS", None)
        if model_cls:
            model_name = model_cls.__name__
    dataset_name = cfg.get("DATASET", {}).get("NAME", "unknown")
    
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    
    # Build model
    print("Building model...")
    try:
        model = build_model_from_cfg(cfg, device)
    except Exception as e:
        print(f"[ERROR] Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create dummy input
    print("Creating dummy input...")
    history, future = create_dummy_input(cfg, device, batch_size)
    print(f"Input shape: history={list(history.shape)}, future={list(future.shape)}")
    
    # Profile
    print("Measuring FLOPs...")
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
    
    # Print summary
    print_flops_summary(results)
    
    # Save CSV
    csv_path = save_flops_csv(results, args.output_dir, model_name, dataset_name)
    print(f"Results saved to: {csv_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
