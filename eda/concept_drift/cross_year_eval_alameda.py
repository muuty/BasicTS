"""Cross-year evaluation for ALAMEDA concept drift Q1 experiments.

Same methodology as SAN_BERNARDINO v3: unmasked MAE, train year scaler.
"""
import sys
import os
import json
import numpy as np
import torch
sys.path.append("/data/pretrainingbasicts")

from baselines.STAEformer.arch import STAEformer
from basicts.scaler import ZScoreScaler

MODEL_PARAM = {
    "num_nodes": 897,
    "in_steps": 12,
    "out_steps": 12,
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

CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_Q1/ALAMEDA_2022_Q1_30_12_12/8e6c0b4865e99debd7fd779f3524c42d/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_Q1/ALAMEDA_2023_Q1_30_12_12/d3d6372d3129236fc8606ff205ff1cc9/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_Q1/ALAMEDA_2024_Q1_30_12_12/392901b793f454934ac4522eeed589d6/STAEformer_best_val_MAE.pt",
}

DATASETS = {
    2022: "datasets/ALAMEDA_2022_Q1",
    2023: "datasets/ALAMEDA_2023_Q1",
    2024: "datasets/ALAMEDA_2024_Q1",
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
DEVICE = "cuda:1"


def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model


def load_data_and_scaler(dataset_dir, year):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)

    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)

    train_data_ch0 = data[:n_train, :, 0]
    mean = float(np.mean(train_data_ch0))
    std = float(np.std(train_data_ch0))

    n_val = int(n_total * 0.2)
    test_start = n_train + n_val
    test_data = data[test_start:]

    return test_data, mean, std, n_total


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    total_len = input_len + output_len
    for i in range(len(data) - total_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+total_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def normalize_input(x, mean, std):
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def evaluate(model, test_x, test_y, mean, std, batch_size=64):
    test_x_norm = normalize_input(test_x, mean, std)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)
    targets = test_y

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return mae, rmse


def main():
    years = [2022, 2023, 2024]
    results = {}

    test_data_cache = {}
    for year in years:
        test_data, mean, std, n_total = load_data_and_scaler(DATASETS[year], year)
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        test_data_cache[year] = {
            "test_x": test_x, "test_y": test_y,
            "mean": mean, "std": std
        }
        print(f"{year} Q1: {len(test_x)} test samples, scaler mean={mean:.2f} std={std:.2f}")

    for train_year in years:
        print(f"\n=== Model trained on {train_year} Q1 ===")
        model = load_model(CHECKPOINTS[train_year])
        train_mean = test_data_cache[train_year]["mean"]
        train_std = test_data_cache[train_year]["std"]

        for test_year in years:
            cache = test_data_cache[test_year]
            mae, rmse = evaluate(
                model, cache["test_x"], cache["test_y"],
                train_mean, train_std
            )
            key = f"train_{train_year}_test_{test_year}"
            results[key] = {"MAE": mae, "RMSE": rmse}
            marker = " (self)" if train_year == test_year else ""
            print(f"  Test on {test_year} Q1: MAE={mae:.4f}, RMSE={rmse:.4f}{marker}")

        del model
        torch.cuda.empty_cache()

    output_path = "eda/concept_drift/cross_year_alameda_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "="*60)
    print("ALAMEDA CROSS-YEAR EVALUATION SUMMARY (Unmasked MAE)")
    print("="*60)
    header = "Train \\ Test"
    print(f"{header:<15} {'2022 Q1':>12} {'2023 Q1':>12} {'2024 Q1':>12}")
    print("-"*51)
    for train_year in years:
        row = f"{train_year} Q1{' ':>7}"
        for test_year in years:
            key = f"train_{train_year}_test_{test_year}"
            mae = results[key]["MAE"]
            marker = "*" if train_year == test_year else " "
            row += f"{mae:>11.4f}{marker}"
        print(row)
    print("-"*51)
    print("* = self-evaluation")


if __name__ == "__main__":
    main()
