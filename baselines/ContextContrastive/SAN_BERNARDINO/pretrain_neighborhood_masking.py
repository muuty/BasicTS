"""
Neighborhood Masking Pre-training on SAN_BERNARDINO Dataset

Novel contribution: Graph-structure aware masking that masks spatially
contiguous regions (k-hop neighborhoods) instead of random nodes.

Key insight: Traffic patterns propagate spatially (congestion spreads along roads).
By masking connected regions, the model learns long-range spatial dependencies
by inferring local patterns from distant observations.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_neighborhood_masking.py', gpus='0')"
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

from ..arch import ConfigurablePretrainModel
from ..loss import get_contrastive_loss

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
MODEL_ARCH = ConfigurablePretrainModel
NUM_NODES = 893

# Encoder parameters
D_MODEL = 64
NUM_LAYERS = 2
NHEAD = 4

# Neighborhood Masking parameters
ADJ_MX_PATH = 'datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl'
MASK_RATIO = 0.3  # Target ~30% of nodes masked
NEIGHBORS_PER_SEED = 10  # Top-k nearest neighbors to mask with each seed
SEED_STRATEGY = 'random'  # 'random' or 'hub'

# Also apply feature masking (best baseline technique)
AUGMENTATION_CONFIG = {
    'neighborhood_masking': {
        'adj_mx_path': ADJ_MX_PATH,
        'mask_ratio': MASK_RATIO,
        'neighbors_per_seed': NEIGHBORS_PER_SEED,
        'seed_strategy': SEED_STRATEGY,
    },
    'feature_masking': {'mask_ratio': 0.3},
}

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
}
NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Neighborhood Masking Pre-training (Graph-Aware Spatial Masking)'
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
CFG.MODEL.NAME = 'Pretrain_NeighborhoodMasking'
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
    'ContextContrastive_neighborhood_mask_3mo',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = get_contrastive_loss(temperature=0.1)

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
