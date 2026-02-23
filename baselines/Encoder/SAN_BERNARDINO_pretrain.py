"""
Pre-training config for ContextAwareSTEncoder.

This config trains the encoder using contrastive learning.
After pre-training, the encoder checkpoint can be used for prediction training.
"""

import os
import sys
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import RepresentationRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from encoder import ContextAwareSTEncoder

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture
MODEL_ARCH = ContextAwareSTEncoder
MODEL_PARAM = {
    "input_dim": 3,          # flow + time features
    "d_model": 64,           # smaller for testing
    "n_nodes": 893,          # SAN_BERNARDINO nodes
    "n_layers": 2,           # fewer layers for testing
    "n_heads": 4,
    "input_len": INPUT_LEN,
    "steps_per_day": 288,
    "dropout": 0.1,
}
NUM_EPOCHS = 10  # short for testing

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'ContextAwareSTEncoder Pre-training Test'
CFG.GPU_NUM = 1

# Use RepresentationRunner
CFG.RUNNER = RepresentationRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
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
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
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
    'pretrain_test',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME.replace('/', '_'), str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)

# Pre-training stage
CFG.TRAIN.STAGE = 'pretrain'

# Loss for pre-training (not used directly, contrastive loss is used instead)
CFG.TRAIN.LOSS = masked_mae

# Optimizer
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
}

# Learning rate scheduler
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "T_max": NUM_EPOCHS,
}

# Data loader
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

############################## Pre-training Configuration ##############################
CFG.PRETRAIN = EasyDict()

# Weak augmentation
CFG.PRETRAIN.WEAK_AUG_STD = 0.1

# Strong augmentation (None for now, will use fallback jittering)
CFG.PRETRAIN.STRONG_AUGMENTATION = None

# Contrastive loss
CFG.PRETRAIN.CONTRASTIVE_LOSS = EasyDict()
CFG.PRETRAIN.CONTRASTIVE_LOSS.TYPE = 'InfoNCELoss'
CFG.PRETRAIN.CONTRASTIVE_LOSS.TEMPERATURE = 0.1

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 5
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.INCIDENT_METADATA_PATH = f"datasets/{DATA_NAME}/incident_metadata_2023.csv"

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
