"""
Disentangled + Temporal-Aware Pretraining (5-feature version)

Input: 5 features (flow, speed, occupancy, tod, dow)
Reconstruction: 3 features (flow, speed, occupancy) - only traffic features

Key insight: tod/dow are deterministic, no need to reconstruct them.
Encoder sees all 5 features for richer temporal context.

Combines:
- z_context: Trained with Temporal-Aware Contrastive Loss
- z_self: Trained with Reconstruction Loss (traffic features only)

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_disentangled_temporal_5feat.py', gpus='0')"
"""
import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.ContextContrastive.arch import DisentangledTemporalModel
from baselines.ContextContrastive.runner import DisentangledTemporalRunner

DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = True  # Override: per-channel normalization for multi-feature reconstruction
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

NUM_NODES = 893
D_MODEL = 64
NUM_LAYERS = 2
NHEAD = 4
NUM_EPOCHS = 30
MASK_RATIO = 0.5

# 5-feature input: flow(0), speed(1), occupancy(2), tod(3), dow(4)
# 3-feature reconstruction: flow, speed, occupancy (traffic features only)
MODEL_PARAM = {
    'input_dim': 5,        # flow, speed, occupancy, tod, dow
    'recon_dim': 3,        # Reconstruct only first 3 (traffic features)
    'd_model': D_MODEL,
    'num_layers': NUM_LAYERS,
    'nhead': NHEAD,
    'dropout': 0.1,
    'fusion': 'concat',
    'mask_ratio': MASK_RATIO,
    # tod/dow are at indices 3, 4 in 5-feature input (or -2, -1)
    'tod_idx': 3,
    'dow_idx': 4,
    # Temporal-aware contrastive params
    'temperature': 0.1,
    'tod_weight': 0.5,
    'dow_weight': 0.5,
    'soft_positive_weight': 0.3,
    # Loss weights
    'contrastive_weight': 1.0,
    'reconstruction_weight': 1.0,
}

PRETRAIN_CONFIG = {
    'contrastive_weight': 1.0,
    'reconstruction_weight': 1.0,
}

CFG = EasyDict()
CFG.DESCRIPTION = 'Disentangled+Temporal Pretrain (5feat): all features input, only traffic reconstructed'
CFG.GPU_NUM = 1
CFG.RUNNER = DisentangledTemporalRunner
CFG.PRETRAIN = PRETRAIN_CONFIG

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    'data_range': (0, 26280),  # 3 months
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
    'target_channel': [0, 1, 2],  # normalize flow, occupancy, speed independently
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'DisentangledTemporal_5feat_3mo'
CFG.MODEL.ARCH = DisentangledTemporalModel
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]  # flow, speed, occupancy, tod, dow
CFG.MODEL.TARGET_FEATURES = [0]  # Predict flow for downstream

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    CFG.MODEL.NAME,
    '_'.join([DATA_NAME.replace('/', '_'), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = None

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}

CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32
