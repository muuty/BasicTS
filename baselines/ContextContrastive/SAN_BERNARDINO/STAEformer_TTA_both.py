"""
Test-Time Adaptation: Combined Mode (Adaptation + Ensemble)

First adapts encoder using self-supervision, then runs ensemble predictions.
Maximum robustness but slowest inference.

TTA Mode: both
- ADAPT_STEPS: 1 gradient step per sample
- ENSEMBLE_RUNS: 3 forward passes (reduced for speed)

Expected: Best worst-k% performance

Based on: spatial_mask freeze (best performing)

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/STAEformer_TTA_both.py', gpus='0')"
"""
import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.runner import RepresentationLearningRunner

############################## Hot Parameters ##############################
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Pre-trained encoder config (must match pre-training)
D_MODEL = 64
NUM_LAYERS = 2
NHEAD = 4

# Downstream LR only (encoder is frozen during training)
DOWNSTREAM_LR = 1e-3

# STAEformer model
MODEL_ARCH = STAEformer
NUM_NODES = 893

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 288,
    "input_dim": D_MODEL + 2,  # encoded (D_MODEL) + tod + dow
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
NUM_EPOCHS = 30

############################## Encoder Configuration (SPATIAL MASKING 15% - FROZEN) ##############################
ENCODER = {
    'type': 'TransformerEncoder',
    'source': 'pretrained',
    'ckpt_path': f'checkpoints/ContextContrastive_spatial_mask_3mo/xtraffic_SAN_BERNARDINO_30_{INPUT_LEN}_{OUTPUT_LEN}/*/Pretrain_SpatialMask_best_val_MAE.pt',
    'input_dim': 3,
    'd_model': D_MODEL,
    'num_layers': NUM_LAYERS,
    'nhead': NHEAD,
    'dropout': 0.1,
    'include_tod_dow': True,
    'freeze': True,  # Frozen during training, adapted at test time
    'lr': 0,
}

############################## TTA Configuration ##############################
TTA = {
    'ENABLED': True,
    'MODE': 'both',               # Use both adaptation + ensemble

    # Ensemble settings (reduced for speed)
    'ENSEMBLE_RUNS': 3,           # Fewer runs since adaptation helps
    'ENSEMBLE_DROPOUT': 0.1,

    # Adaptation settings
    'ADAPT_STEPS': 1,             # Gradient steps per test sample
    'ADAPT_LR': 1e-4,             # Small LR to avoid overfitting
    'MASK_RATIO': 0.15,           # Node masking ratio
    'ADAPT_LAYERS': 'all',        # Which layers to adapt
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'TTA Both: Spatial Mask Freeze + Adaptation + Ensemble at Test Time'
CFG.GPU_NUM = 1
CFG.RUNNER = RepresentationLearningRunner
CFG.ENCODER = ENCODER
CFG.TTA = TTA

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    'data_range': (0, 26280),  # 3 months only
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'STAEformer_TTA_Both'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'ContextContrastive_TTA_both_3mo',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": DOWNSTREAM_LR, "weight_decay": 0.0003}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.1}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.INCIDENT_METADATA_PATH = f"datasets/{DATA_NAME}/incident_metadata_2023.csv"

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
