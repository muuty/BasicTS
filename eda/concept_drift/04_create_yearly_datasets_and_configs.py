#!/usr/bin/env python3
"""
Create per-year datasets and STAEformer configs for cross-year experiments.
Also includes a quick persistence-model baseline (no training needed).
"""

import numpy as np
import json
import os
import shutil
from pathlib import Path

STEPS_PER_DAY = 288
BASE_DIR = Path("/data/pretrainingbasicts")
SRC_3Y = BASE_DIR / "datasets" / "SAN_BERNARDINO_3Y"

# Load 3-year desc
with open(SRC_3Y / "desc.json") as f:
    desc_3y = json.load(f)

year_boundaries = {int(k): v for k, v in desc_3y["year_boundaries"].items()}
total_steps = desc_3y["num_time_steps"]

# Load the full 3-year data
print("Loading 3-year data...")
data_3y = np.memmap(SRC_3Y / "data.dat", dtype="float32", mode="r",
                     shape=tuple(desc_3y["shape"]))

# ──────────────────────────────────────────
# 1. Create per-year datasets
# ──────────────────────────────────────────
for year in [2022, 2023, 2024]:
    start, end = year_boundaries[year]
    year_data = data_3y[start:end]
    n_steps = end - start

    out_dir = BASE_DIR / "datasets" / f"SAN_BERNARDINO_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save data.dat
    fp = np.memmap(out_dir / "data.dat", dtype="float32", mode="w+", shape=year_data.shape)
    fp[:] = year_data[:]
    fp.flush()
    del fp

    # Copy adj
    shutil.copy(SRC_3Y / "adj_mx.pkl", out_dir / "adj_mx.pkl")

    # Save desc.json
    desc = {
        "name": f"SAN_BERNARDINO_{year}",
        "domain": "traffic flow",
        "shape": list(year_data.shape),
        "num_time_steps": int(n_steps),
        "num_nodes": 893,
        "num_features": 5,
        "feature_description": ["flow", "occupancy", "speed", "time of day", "day of week"],
        "has_graph": False,
        "frequency (minutes)": 5,
        "steps_per_day": STEPS_PER_DAY,
        "county": "San Bernardino",
        "year": year,
        "regular_settings": {
            "INPUT_LEN": 12,
            "OUTPUT_LEN": 12,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": True,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0
        }
    }
    with open(out_dir / "desc.json", "w") as f:
        json.dump(desc, f, indent=4)

    print(f"Created {out_dir.name}: shape={year_data.shape}, {n_steps/STEPS_PER_DAY:.0f} days")

# ──────────────────────────────────────────
# 2. Quick persistence model baseline
# ──────────────────────────────────────────
print("\n" + "="*70)
print("PERSISTENCE MODEL BASELINE (last-value predictor)")
print("="*70)
print("Predicts: output = last input value (for each horizon)")
print("Test split: last 20% of each year\n")

for year in [2022, 2023, 2024]:
    start, end = year_boundaries[year]
    year_data = data_3y[start:end, :, 0]  # flow only, (T, N)
    n_steps = end - start

    # Test split: last 20%
    test_start = int(n_steps * 0.8)
    test_data = year_data[test_start:]  # (T_test, N)

    input_len = 12
    output_len = 12

    maes = []
    for i in range(len(test_data) - input_len - output_len + 1):
        last_val = test_data[i + input_len - 1]  # (N,) last input
        future = test_data[i + input_len : i + input_len + output_len]  # (12, N)
        # masked MAE (exclude zeros)
        mask = future != 0
        if mask.sum() > 0:
            err = np.abs(future - last_val[None, :]) * mask
            mae = err.sum() / mask.sum()
            maes.append(mae)

    avg_mae = np.mean(maes)
    print(f"{year}: Persistence MAE = {avg_mae:.4f} (n_samples={len(maes)})")

# ──────────────────────────────────────────
# 3. Cross-year persistence: train-year scaler, test on other year
# ──────────────────────────────────────────
print("\n" + "="*70)
print("CROSS-YEAR PERSISTENCE MODEL")
print("="*70)
print("Same persistence model, but compare test-set MAE across years")
print("(Persistence doesn't actually learn, so this measures test difficulty)\n")

# Per-year test MAE breakdown by functional/dead/etc
# Load sensor categories
dead_idx = np.load(BASE_DIR / "datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy")
func_idx = np.load(BASE_DIR / "datasets/xtraffic/SAN_BERNARDINO/functional_node_indices.npy")

for year in [2022, 2023, 2024]:
    start, end = year_boundaries[year]
    year_data = data_3y[start:end, :, 0]  # flow only
    n_steps = end - start
    test_start = int(n_steps * 0.8)
    test_data = year_data[test_start:]

    input_len = 12
    output_len = 12

    per_node_maes = np.zeros(893)
    per_node_counts = np.zeros(893)

    for i in range(0, len(test_data) - input_len - output_len + 1, 12):  # stride=12 for speed
        last_val = test_data[i + input_len - 1]
        future = test_data[i + input_len : i + input_len + output_len]
        mask = future != 0
        err = np.abs(future - last_val[None, :])
        for n in range(893):
            m = mask[:, n]
            if m.sum() > 0:
                per_node_maes[n] += (err[:, n] * m).sum()
                per_node_counts[n] += m.sum()

    per_node_maes = np.where(per_node_counts > 0, per_node_maes / per_node_counts, 0)

    overall = per_node_maes[per_node_counts > 0].mean()
    func_mae = per_node_maes[func_idx].mean()
    dead_mae = per_node_maes[dead_idx].mean()

    print(f"{year}: Overall={overall:.2f}, Functional={func_mae:.2f}, Dead={dead_mae:.2f}")

# ──────────────────────────────────────────
# 4. Generate config files
# ──────────────────────────────────────────
print("\n" + "="*70)
print("GENERATING CONFIG FILES")
print("="*70)

config_template = '''import os
import sys
import torch
import numpy as np
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.STAEformer.arch import STAEformer

############################## Configuration ##############################
DATA_NAME = '{dataset_name}'
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NUM_NODES = 893

MODEL_PARAM = {{
    "num_nodes": NUM_NODES,
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
}}

CFG = EasyDict()
CFG.DESCRIPTION = '{description}'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
}})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': True,
    'rescale': True,
}})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = STAEformer.__name__
CFG.MODEL.ARCH = STAEformer
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
}})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = 0.0

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = 30
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'ConceptDrift_STAEformer',
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {{
    "lr": 0.001,
    "weight_decay": 0.0003,
}}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {{
    "milestones": [20, 25],
    "gamma": 0.1
}}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
'''

configs_dir = BASE_DIR / "baselines" / "STAEformer"
experiments = [
    ("SAN_BERNARDINO_2022", "concept_drift_2022", "STAEformer trained on 2022 data only"),
    ("SAN_BERNARDINO_2023", "concept_drift_2023", "STAEformer trained on 2023 data only"),
    ("SAN_BERNARDINO_2024", "concept_drift_2024", "STAEformer trained on 2024 data only"),
    ("SAN_BERNARDINO_3Y", "concept_drift_3Y", "STAEformer trained on all 3 years (2022-2024)"),
]

for dataset_name, config_name, description in experiments:
    config_path = configs_dir / f"{config_name}.py"
    config_content = config_template.format(
        dataset_name=dataset_name,
        description=description,
    )
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created: {config_path}")

print("\n✅ All datasets and configs created!")
print("\nTo queue experiments:")
print("  python experiments/ray_queue.py add \\")
for _, config_name, _ in experiments:
    print(f"    baselines/STAEformer/{config_name}.py \\")
