"""
Ablation Study: Temporal-Aware Negative Sampling Pre-training

Improves upon standard contrastive learning by using temporal context
(time_of_day, day_of_week) for smarter negative sampling:
- Same tod/dow samples are treated as soft positives
- Different tod/dow samples are treated as hard negatives

This encourages learning representations that distinguish different
temporal patterns (e.g., rush hour vs. midnight, weekday vs. weekend).

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_temporal_aware_negative.py', gpus='1')"
"""
import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from ..arch import TemporalAwarePretrainModel
from ..loss import get_temporal_aware_contrastive_loss

############################## Hot Parameters ##############################
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture
MODEL_ARCH = TemporalAwarePretrainModel
NUM_NODES = 893

# Encoder parameters
D_MODEL = 64
NUM_LAYERS = 2
NHEAD = 4

# Temporal-Aware Contrastive Learning parameters
TEMPERATURE = 0.1
TOD_WEIGHT = 0.5       # Weight for time-of-day in negative sampling
DOW_WEIGHT = 0.5       # Weight for day-of-week in negative sampling
SOFT_POSITIVE_WEIGHT = 0.3  # Weight for same-context samples (lower = softer)

# Augmentation: Feature Masking (same as best baseline)
AUGMENTATION_CONFIG = {
    'feature_masking': {'mask_ratio': 0.3},
}

# Feature indices in input data (after FORWARD_FEATURES selection)
# FORWARD_FEATURES = [0, 3, 4] -> [flow, tod, dow]
# So in the selected features: tod_idx=1, dow_idx=2
TOD_FEATURE_IDX = 1
DOW_FEATURE_IDX = 2

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "input_dim": 3,  # flow, tod, dow
    "output_dim": 1,
    "d_model": D_MODEL,
    "num_layers": NUM_LAYERS,
    "nhead": NHEAD,
    "dropout": 0.1,
    "augmentation_config": AUGMENTATION_CONFIG,
    "tod_feature_idx": TOD_FEATURE_IDX,
    "dow_feature_idx": DOW_FEATURE_IDX,
}
NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Ablation: Temporal-Aware Negative Sampling Pre-training'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

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
CFG.MODEL.NAME = 'Pretrain_TemporalAwareNegative'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 3, 4]  # flow, tod, dow
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
    'ContextContrastive_temporal_aware_neg_3mo',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = get_temporal_aware_contrastive_loss(
    temperature=TEMPERATURE,
    tod_weight=TOD_WEIGHT,
    dow_weight=DOW_WEIGHT,
    soft_positive_weight=SOFT_POSITIVE_WEIGHT,
)

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

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
