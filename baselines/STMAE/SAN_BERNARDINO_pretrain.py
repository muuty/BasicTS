"""
STMAE Pre-training Configuration for SAN_BERNARDINO Dataset.

This configuration sets up STMAE for pre-training with:
- Feature masking (temporal patches)
- Structure masking (graph edges via random walks)
- AGCRN encoder backbone
- Combined reconstruction losses
"""

import os
import sys
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.STMAE.arch import STMAE
from baselines.STMAE.runner import STMAEPretrainRunner
from baselines.STMAE.loss import stmae_loss

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # 12
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # 12
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture
MODEL_ARCH = STMAE
NUM_NODES = 893

# AGCRN backbone parameters
AGCRN_PARAMS = {
    "num_nodes": NUM_NODES,
    "input_dim": 1,  # Only flow for encoder
    "rnn_units": 64,
    "num_layers": 2,
    "cheb_k": 2,
    "embed_dim": 10,
}

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_dim": 1,  # Only flow
    "hidden_dim": 64,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    # Backbone configuration (use string for pickle safety)
    "backbone_class": "AGCRN",
    "backbone_params": AGCRN_PARAMS,
    # Masking configuration
    "mask_f_ratio": 0.5,  # Mask 50% of temporal patches
    "mask_s_ratio": 0.3,  # Mask 30% of edges
    "patch_length": 1,
    "walks_per_node": 10,
    "walk_length": 20,
    # Decoder configuration
    "stru_dec_dropout": 0.0,
    "stru_dec_proj": False,
    # Loss weights
    "sl_weight": 1.0,  # Structure loss weight
    "fl_weight": 1.0,  # Feature loss weight
    # Node embeddings
    "embed_dim": 10,
}

NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STMAE Pre-training on SAN_BERNARDINO'
CFG.GPU_NUM = 1

# Runner
CFG.RUNNER = STMAEPretrainRunner

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
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]  # flow only
CFG.MODEL.TARGET_FEATURES = [0]  # flow only

# STMAE-specific model configuration
CFG.MODEL.MASK_F_RATIO = 0.5
CFG.MODEL.MASK_S_RATIO = 0.3
CFG.MODEL.PATCH_LENGTH = 1
CFG.MODEL.EPOCH_WISE_MASK = False  # Generate new masks each batch

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'STMAE_pretrain',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)

# Loss function
CFG.TRAIN.LOSS = stmae_loss
CFG.TRAIN.SL_WEIGHT = 1.0  # Structure loss weight
CFG.TRAIN.FL_WEIGHT = 1.0  # Feature loss weight

# Optimizer
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}

# Gradient clipping (prevents gradient explosion -> NaN -> MAE=0)
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

# Learning rate scheduler
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20, 25],
    "gamma": 0.1
}

# Data loader
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
CFG.EVAL.HORIZONS = []  # No horizon-specific metrics for pre-training
CFG.EVAL.USE_GPU = True
