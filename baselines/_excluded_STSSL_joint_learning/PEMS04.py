"""
ST-SSL Configuration for PEMS04 Dataset.

ST-SSL: Self-Supervised Learning for Spatio-Temporal Traffic Prediction
Paper: https://arxiv.org/abs/2212.04475
"""
import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import STSSL
from .runner import STSSLRunner

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS04'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture
MODEL_ARCH = STSSL
NUM_NODES = 307

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "input_dim": 1,  # Only use traffic flow feature for ST encoder
    "output_dim": 1,
    "d_model": 64,
    "dropout": 0.1,
    # Contrastive learning params
    "nmb_prototype": 10,
    "shm_temp": 0.5,
    "aug_percent": 0.2,
    # Loss weights: [pred_loss, temporal_loss, spatial_loss]
    "loss_weights": [1.0, 0.1, 0.1],
    # Training mode: 'pretrain', 'finetune', 'end2end'
    "mode": "end2end",
    # Batch size for temporal hetero model
    "batch_size": 16,
    # Encoder params
    "Kt": 3,
    "Ks": 3,
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'ST-SSL model for PEMS04 dataset'
CFG.GPU_NUM = 1

# Runner
CFG.RUNNER = STSSLRunner

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

############################## Graph Configuration ##############################
CFG.GRAPH = EasyDict()
CFG.GRAPH.adj_path = f'datasets/{DATA_NAME}/adj_mx.pkl'

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
CFG.MODEL.FORWARD_FEATURES = [0]  # Only use traffic flow
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
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}

# Learning rate scheduler
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50, 80],
    "gamma": 0.5
}

# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
