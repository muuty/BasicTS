"""
STMAE Fine-tuning Configuration for SAN_BERNARDINO Dataset.

This configuration fine-tunes STMAE with pre-trained encoder for:
- Traffic forecasting on SAN_BERNARDINO dataset
- Using pre-trained weights from SAN_BERNARDINO_pretrain.py
- STAEformer backbone for downstream forecasting
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

from baselines.STMAE.arch import STMAE
from baselines.STMAE.runner import STMAEFinetuneRunner
from baselines.STAEformer.arch import STAEformer

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
MODEL_ARCH = STAEformer
NUM_NODES = 893

# Pre-trained model path (will be updated after pretrain completes)
# The runner saves pretrained_encoder.pt at the end of training
PRE_TRAINED_STMAE_PATH = os.path.join(
    'checkpoints',
    'STMAE_pretrain',
    f'xtraffic_SAN_BERNARDINO_30_{INPUT_LEN}_{OUTPUT_LEN}',
    '*',
    'pretrained_encoder.pt'
)

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 288,
    "input_dim": 3,  # flow + tod + dow
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
    # Fine-tuning configuration
    "pretrained_path": PRE_TRAINED_STMAE_PATH,
    "freeze_encoder": False,  # Fine-tune the encoder
}

NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STMAE Fine-tuning with STAEformer on SAN_BERNARDINO'
CFG.GPU_NUM = 1

# Runner
CFG.RUNNER = STMAEFinetuneRunner

############################## Fine-tuning Configuration ##############################
CFG.FINETUNE = EasyDict()
CFG.FINETUNE.PRETRAINED_PATH = PRE_TRAINED_STMAE_PATH
CFG.FINETUNE.FREEZE_ENCODER = False
CFG.FINETUNE.ENCODER_LR = 1e-5  # Lower learning rate for encoder
CFG.FINETUNE.DOWNSTREAM_LR = 1e-3  # Standard learning rate for downstream

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
CFG.MODEL.NAME = 'STMAE_STAEformer'
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
    'STMAE_finetune',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)

# Loss function
CFG.TRAIN.LOSS = masked_mae

# Optimizer (will be configured with separate LRs in runner)
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,  # Default LR for downstream (will be overridden)
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
