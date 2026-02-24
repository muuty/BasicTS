"""Pattern Bank Expanding Sensor Experiment v2.

Clean cold-start: models trained on 700/800 nodes, expand to 893.
Compares Baseline (standard embedding) vs PB (pattern bank) adaptation.

Methods:
  A: emb_all          - Baseline: FT all adaptive_embedding
  B: emb_new_only     - Baseline: FT new node embeddings only
  C: pb_weights_all   - PB: FT all node_weights (prototypes frozen)
  D: pb_weights_new   - PB: FT new node_weights only (prototypes frozen)
  E: pb_both          - PB: FT pattern_bank + all node_weights

Usage:
    source ~/.conda/etc/profile.d/conda.sh && conda activate basicts
    python -u eda/concept_drift/pattern_bank_expanding.py
"""
import sys
import os
import json
import glob
import copy
import numpy as np
import torch
import torch.nn as nn

sys.path.append("/data/pretrainingbasicts")
from baselines.STAEformer.arch import STAEformer

# ============================================================
# Config
# ============================================================
DEVICE = "cuda:1"
TOTAL_NODES = 893
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
BATCH_SIZE = 16
FT_EPOCHS = 10
FT_LR = 0.001
FT_HOURS = [3, 12, 24, 168]  # 3h, 12h, 1d, 7d

NODE_DIR = "datasets/xtraffic/SAN_BERNARDINO"

BASE_MODEL_PARAM = {
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 288,
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
}

YEAR_PAIRS = [
    {"source": 2022, "target": 2023, "nodes_file": "expanding_nodes_2022.npy"},
    {"source": 2023, "target": 2024, "nodes_file": "expanding_nodes_2023.npy"},
]

CKPT_DIRS = {
    "baseline": "checkpoints/Expanding_Baseline",
    "pb": "checkpoints/Expanding_PB",
}

OUTPUT_PATH = "eda/concept_drift/pattern_bank_expanding_results.json"


# ============================================================
# Data utilities
# ============================================================

def load_data(year):
    """Load Q1 dataset and compute ZScore stats from training split."""
    dataset_dir = f"datasets/SAN_BERNARDINO_{year}_Q1"
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"),
                     dtype="float32", mode="r").reshape(shape)
    n_train = int(shape[0] * TRAIN_RATIO)
    mean = float(np.mean(data[:n_train, :, 0]))
    std = float(np.std(data[:n_train, :, 0]))
    return data, mean, std


def create_samples(data):
    """Create sliding window samples. Returns (x, y_raw)."""
    xs, ys = [], []
    for i in range(len(data) - INPUT_LEN - OUTPUT_LEN + 1):
        xs.append(data[i:i + INPUT_LEN])
        ys.append(data[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN, :, 0:1])
    return np.array(xs), np.array(ys)


def get_test_data(year, mean, std):
    """Get test split samples."""
    data, _, _ = load_data(year)
    n_total = data.shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * 0.2)
    test_data = data[n_train + n_val:]
    return create_samples(test_data)


def get_fewshot_data(year, hours):
    """Get first N hours of target year data."""
    data, _, _ = load_data(year)
    n_steps = hours * 12  # 5-min intervals
    fs_data = data[:n_steps]
    return create_samples(fs_data)


# ============================================================
# Instance norm helpers
# ============================================================

def apply_instance_norm(bx):
    """Instance-normalize flow (ch0) per window. Input already ZScore-normalized."""
    flow = bx[:, :, :, 0]
    mu = flow.mean(dim=1, keepdim=True)
    sigma = flow.std(dim=1, keepdim=True) + 1e-5
    bx_out = bx.clone()
    bx_out[:, :, :, 0] = (flow - mu) / sigma
    return bx_out, mu, sigma


def denorm_prediction(pred, mu, sigma, global_mean, global_std):
    """De-instance-norm then de-ZScore. pred: (B, T, N, 1)."""
    pred = pred * sigma.unsqueeze(-1) + mu.unsqueeze(-1)  # de-instance-norm
    pred = pred * global_std + global_mean  # de-ZScore
    return pred


# ============================================================
# Model expansion
# ============================================================

def find_checkpoint(ckpt_base, data_name):
    """Find best_val checkpoint under ckpt_base/data_name_30_12_12/*/."""
    pattern = os.path.join(ckpt_base, f"{data_name}_30_12_12", "*",
                           "STAEformer_best_val_MAE.pt")
    matches = glob.glob(pattern)
    assert len(matches) == 1, f"Expected 1 checkpoint, found {len(matches)}: {pattern}"
    return matches[0]


def expand_model(ckpt_path, source_nodes, num_patterns=0):
    """Load trained model and expand from len(source_nodes) to TOTAL_NODES.

    Args:
        ckpt_path: Path to trained checkpoint
        source_nodes: List of node indices the model was trained on
        num_patterns: 0 for baseline, >0 for PB

    Returns:
        model: Expanded 893-node model on DEVICE
        existing_idx: indices of existing nodes in 893-node system
        new_idx: indices of new nodes in 893-node system
    """
    n_source = len(source_nodes)
    existing_idx = sorted(source_nodes)
    new_idx = sorted(set(range(TOTAL_NODES)) - set(existing_idx))

    # Load source model
    src_param = {**BASE_MODEL_PARAM, "num_nodes": n_source, "num_patterns": num_patterns}
    src_model = STAEformer(**src_param)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src_model.load_state_dict(ckpt["model_state_dict"])

    # Create target model (893 nodes)
    tgt_param = {**BASE_MODEL_PARAM, "num_nodes": TOTAL_NODES, "num_patterns": num_patterns}
    tgt_model = STAEformer(**tgt_param)

    # Copy all weights from source
    src_sd = src_model.state_dict()
    tgt_sd = tgt_model.state_dict()

    for key in tgt_sd:
        if key in src_sd and src_sd[key].shape == tgt_sd[key].shape:
            tgt_sd[key] = src_sd[key].clone()

    # Expand node-specific parameters
    if num_patterns == 0:
        # Baseline: adaptive_embedding (T, N_src, D) → (T, N_total, D)
        old_emb = src_sd["encoder.adaptive_embedding"]  # (T, N_src, D)
        new_emb = torch.zeros(old_emb.shape[0], TOTAL_NODES, old_emb.shape[2])
        for i, node_id in enumerate(existing_idx):
            new_emb[:, node_id, :] = old_emb[:, i, :]
        tgt_sd["encoder.adaptive_embedding"] = new_emb
    else:
        # PB: node_weights (N_src, K) → (N_total, K), pattern_bank unchanged
        old_w = src_sd["encoder.adaptive_embedding.node_weights"]  # (N_src, K)
        new_w = torch.zeros(TOTAL_NODES, old_w.shape[1])
        for i, node_id in enumerate(existing_idx):
            new_w[node_id, :] = old_w[i, :]
        tgt_sd["encoder.adaptive_embedding.node_weights"] = new_w
        # pattern_bank already copied (same shape, node-agnostic)
        tgt_sd["encoder.adaptive_embedding.pattern_bank"] = src_sd["encoder.adaptive_embedding.pattern_bank"].clone()

    tgt_model.load_state_dict(tgt_sd)
    tgt_model = tgt_model.to(DEVICE)
    tgt_model.eval()
    return tgt_model, existing_idx, new_idx


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, test_x, test_y, mean, std, existing_idx, new_idx):
    """Evaluate MAE per group. Returns dict with all/existing/new MAE."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x), 64):
            bx = torch.FloatTensor(test_x[i:i + 64]).to(DEVICE)
            # ZScore normalize flow
            bx[:, :, :, 0] = (bx[:, :, :, 0] - mean) / std
            # Instance norm
            bx_n, mu, sigma = apply_instance_norm(bx)
            out = model(bx_n, None, 0, 0, False)["prediction"]
            pred = denorm_prediction(out, mu, sigma, mean, std)
            all_preds.append(pred.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N_samples, T_out, 893, 1)
    targets = test_y  # (N_samples, T_out, 893, 1)

    mae_all = float(np.mean(np.abs(preds - targets)))
    mae_exist = float(np.mean(np.abs(preds[:, :, existing_idx] - targets[:, :, existing_idx])))
    mae_new = float(np.mean(np.abs(preds[:, :, new_idx] - targets[:, :, new_idx])))
    return {"all": mae_all, "existing": mae_exist, "new": mae_new}


# ============================================================
# Fine-tuning
# ============================================================

def _train_loop(model, params, train_x, train_y, mean, std, grad_hooks=None):
    """Generic training loop. Returns model after FT_EPOCHS."""
    model.train()
    optimizer = torch.optim.Adam(params, lr=FT_LR, weight_decay=0)

    for epoch in range(FT_EPOCHS):
        indices = np.random.permutation(len(train_x))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            bx = torch.FloatTensor(train_x[batch_idx]).to(DEVICE)
            by = torch.FloatTensor(train_y[batch_idx]).to(DEVICE)

            # ZScore normalize flow
            bx[:, :, :, 0] = (bx[:, :, :, 0] - mean) / std

            bx_n, mu, sigma = apply_instance_norm(bx)
            pred = model(bx_n, None, 0, 0, True)["prediction"]
            pred = denorm_prediction(pred, mu, sigma, mean, std)

            loss = nn.L1Loss()(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{FT_EPOCHS} loss={epoch_loss/n_batches:.4f}")

    # Clean up grad hooks
    if grad_hooks:
        for h in grad_hooks:
            h.remove()

    model.eval()
    return model


def finetune_emb_all(model, train_x, train_y, mean, std, existing_idx, new_idx):
    """Method A: FT all adaptive_embedding (baseline)."""
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.adaptive_embedding.requires_grad = True
    return _train_loop(model, [model.encoder.adaptive_embedding], train_x, train_y, mean, std)


def finetune_emb_new_only(model, train_x, train_y, mean, std, existing_idx, new_idx):
    """Method B: FT new node embeddings only (baseline)."""
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.adaptive_embedding.requires_grad = True

    new_set = set(new_idx)
    def mask_hook(grad):
        mask = torch.zeros_like(grad)
        mask[:, new_idx, :] = 1.0
        return grad * mask

    h = model.encoder.adaptive_embedding.register_hook(mask_hook)
    return _train_loop(model, [model.encoder.adaptive_embedding], train_x, train_y, mean, std, grad_hooks=[h])


def finetune_pb_weights_all(model, train_x, train_y, mean, std, existing_idx, new_idx):
    """Method C: FT all node_weights, prototypes frozen (PB)."""
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.adaptive_embedding.node_weights.requires_grad = True
    return _train_loop(model, [model.encoder.adaptive_embedding.node_weights], train_x, train_y, mean, std)


def finetune_pb_weights_new(model, train_x, train_y, mean, std, existing_idx, new_idx):
    """Method D: FT new node_weights only, prototypes frozen (PB)."""
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.adaptive_embedding.node_weights.requires_grad = True

    def mask_hook(grad):
        mask = torch.zeros_like(grad)
        mask[new_idx, :] = 1.0
        return grad * mask

    h = model.encoder.adaptive_embedding.node_weights.register_hook(mask_hook)
    return _train_loop(model, [model.encoder.adaptive_embedding.node_weights], train_x, train_y, mean, std, grad_hooks=[h])


def finetune_pb_both(model, train_x, train_y, mean, std, existing_idx, new_idx):
    """Method E: FT pattern_bank + all node_weights (PB)."""
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.adaptive_embedding.node_weights.requires_grad = True
    model.encoder.adaptive_embedding.pattern_bank.requires_grad = True
    params = [
        model.encoder.adaptive_embedding.node_weights,
        model.encoder.adaptive_embedding.pattern_bank,
    ]
    return _train_loop(model, params, train_x, train_y, mean, std)


# Method registry: (method_name, model_type, finetune_fn, trainable_params_desc)
METHODS = [
    ("A_emb_all",        "baseline", finetune_emb_all,         "24 x N"),
    ("B_emb_new_only",   "baseline", finetune_emb_new_only,    "24 x N_new"),
    ("C_pb_weights_all", "pb",       finetune_pb_weights_all,  "8 x N"),
    ("D_pb_weights_new", "pb",       finetune_pb_weights_new,  "8 x N_new"),
    ("E_pb_both",        "pb",       finetune_pb_both,         "8xN + 8x12x24"),
]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PATTERN BANK EXPANDING SENSOR EXPERIMENT v2")
    print("=" * 70)

    results = {}

    for pair in YEAR_PAIRS:
        src_year = pair["source"]
        tgt_year = pair["target"]
        nodes_file = pair["nodes_file"]
        pair_key = f"{src_year}_{tgt_year}"

        print(f"\n{'=' * 60}")
        print(f"  YEAR PAIR: {src_year} -> {tgt_year}")
        print(f"{'=' * 60}")

        # Load source nodes
        source_nodes = np.load(os.path.join(NODE_DIR, nodes_file)).tolist()
        n_source = len(source_nodes)
        print(f"  Source nodes: {n_source}, New nodes: {TOTAL_NODES - n_source}")

        # Load source year ZScore stats (what the model learned with)
        _, src_mean, src_std = load_data(src_year)
        print(f"  Source ZScore: mean={src_mean:.2f}, std={src_std:.2f}")

        # Load test data from target year
        test_x, test_y = get_test_data(tgt_year, src_mean, src_std)
        print(f"  Test samples: {len(test_x)}")

        data_name = f"SAN_BERNARDINO_{src_year}_Q1"

        for model_type in ["baseline", "pb"]:
            num_patterns = 8 if model_type == "pb" else 0
            ckpt_path = find_checkpoint(CKPT_DIRS[model_type], data_name)
            print(f"\n  --- Model: {model_type} (from {ckpt_path}) ---")

            # Expand model
            model, existing_idx, new_idx = expand_model(
                ckpt_path, source_nodes, num_patterns)
            n_new = len(new_idx)

            # Cold-start MAE@0
            mae_cold = evaluate(model, test_x, test_y, src_mean, src_std,
                                existing_idx, new_idx)
            rkey = f"{model_type}_cold_{pair_key}"
            results[rkey] = mae_cold
            print(f"    Cold-start: all={mae_cold['all']:.2f} "
                  f"exist={mae_cold['existing']:.2f} new={mae_cold['new']:.2f}")
            del model; torch.cuda.empty_cache()

            # Fine-tune with each applicable method x budget
            applicable = [(n, fn, desc) for n, mt, fn, desc in METHODS if mt == model_type]

            for method_name, ft_fn, param_desc in applicable:
                print(f"\n    Method {method_name} (trainable: {param_desc})")

                for hours in FT_HOURS:
                    h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"

                    # Fresh model for each method x budget
                    model, existing_idx, new_idx = expand_model(
                        ckpt_path, source_nodes, num_patterns)

                    # Get few-shot data
                    ft_x, ft_y = get_fewshot_data(tgt_year, hours)
                    print(f"      {h_label} ({len(ft_x)} samples) ...", end=" ", flush=True)

                    # Record existing MAE before FT (for forgetting)
                    mae_before = evaluate(model, test_x, test_y, src_mean, src_std,
                                          existing_idx, new_idx)

                    # Fine-tune
                    model = ft_fn(model, ft_x, ft_y, src_mean, src_std,
                                  existing_idx, new_idx)

                    # Evaluate
                    mae_after = evaluate(model, test_x, test_y, src_mean, src_std,
                                         existing_idx, new_idx)

                    forgetting = mae_after["existing"] - mae_before["existing"]

                    rkey = f"{method_name}_{h_label}_{pair_key}"
                    results[rkey] = {
                        **mae_after,
                        "forgetting": forgetting,
                        "n_trainable": param_desc.replace("N", str(TOTAL_NODES)).replace("N_new", str(n_new)),
                    }

                    print(f"all={mae_after['all']:.2f} exist={mae_after['existing']:.2f} "
                          f"new={mae_after['new']:.2f} forget={forgetting:+.2f}")

                    del model; torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for pair in YEAR_PAIRS:
        pk = f"{pair['source']}_{pair['target']}"
        source_nodes = np.load(os.path.join(NODE_DIR, pair["nodes_file"]))
        n_new = TOTAL_NODES - len(source_nodes)
        print(f"\n  {pair['source']} -> {pair['target']} "
              f"({len(source_nodes)} existing, {n_new} new)")

        # Cold-start
        for mt in ["baseline", "pb"]:
            r = results.get(f"{mt}_cold_{pk}", {})
            print(f"    {mt:>10} cold: all={r.get('all',0):.2f} "
                  f"exist={r.get('existing',0):.2f} new={r.get('new',0):.2f}")

        # Methods
        print(f"\n    {'Method':<22} {'Budget':>6} {'All':>7} {'Exist':>7} "
              f"{'New':>7} {'Forget':>8}")
        print(f"    {'-'*60}")

        for method_name, mt, _, _ in METHODS:
            for hours in FT_HOURS:
                h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
                r = results.get(f"{method_name}_{h_label}_{pk}", {})
                if r:
                    print(f"    {method_name:<22} {h_label:>6} {r['all']:>7.2f} "
                          f"{r['existing']:>7.2f} {r['new']:>7.2f} "
                          f"{r.get('forgetting',0):>+8.2f}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
