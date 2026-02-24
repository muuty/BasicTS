"""
GPT-ST Fine-tuning Configuration for SAN_BERNARDINO Dataset.

This configuration fine-tunes GPT-ST with pre-trained encoder for:
- Traffic forecasting on SAN_BERNARDINO dataset
- Using pre-trained weights from SAN_BERNARDINO_pretrain.py
- 3 months of data (matching pre-training)
"""

import os
import sys
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.GPTST.arch import GPTSTForForecasting
from baselines.GPTST.runner import GPTSTFinetuneRunner
from baselines.GPTST.loss import gptst_finetune_loss

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
MODEL_ARCH = GPTSTForForecasting
NUM_NODES = 893

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "input_dim": 3,  # flow + tod + dow
    "output_dim": 1,
    "input_base_dim": 1,  # Only flow for forecasting
    # GPT-ST specific parameters
    "hidden_dim": 64,
    "embed_dim": 16,
    "embed_dim_spa": 8,
    "HS": 4,  # Spatial hyperedge heads
    "HT": 4,  # Temporal hyperedge heads
    "HT_Tem": 4,  # Temporal hypergraph heads
    "num_route": 3,  # Capsule routing iterations
    # Fine-tuning parameters
    "mode": "finetune",
    "pretrained_path": "checkpoints/GPTSTForForecasting/xtraffic/SAN_BERNARDINO_30_12_12_pretrain/3b07cbda17b88b8a5eab0fdd7d0c8f0c/GPTSTForForecasting_best_val_MAE.pt",
    "freeze_encoder": False,  # Fine-tune encoder
}

NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'GPT-ST Fine-tuning on SAN_BERNARDINO'
CFG.GPU_NUM = 1

# Runner
CFG.RUNNER = GPTSTFinetuneRunner

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
CFG.MODEL.FORWARD_FEATURES = [0, 3, 4]  # flow, tod, dow
CFG.MODEL.TARGET_FEATURES = [0]  # flow only

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
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN), 'finetune'])
)

# Loss function
CFG.TRAIN.LOSS = gptst_finetune_loss
CFG.TRAIN.LOSS_ARGS = {
    'null_val': NULL_VAL,
}

# Optimizer
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0003,
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
CFG.TEST.INCIDENT_METADATA_PATH = f"datasets/{DATA_NAME}/incident_metadata_2023.csv"

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
