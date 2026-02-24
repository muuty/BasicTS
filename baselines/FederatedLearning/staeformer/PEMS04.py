#!/usr/bin/env python3
"""
Federated Learning + STAEformer Configuration for PEMS04 Dataset

Usage:
    python experiments/train.py -c baselines/FederatedLearning/STAEformer-PEMS04.py -g 0
"""

import os
import sys
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import FederatedLearningRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from ..arch import FederatedLearningModel
from ...STAEformer.arch import STAEformer

############################## Hot Parameters ##############################
DATA_NAME = 'PEMS04'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Dataset specific
NUM_NODES = 307
NUM_EPOCHS = 30
LR_MILESTONES = [20, 25]
BATCH_SIZE = 16

# Federated Learning 설정
NUM_CLIENTS = 10
PARTITION_METHOD = 'random'
AGGREGATOR_TYPE = 'fedavg'
AGGREGATOR_PARAMS = {}
NUM_ROUNDS = NUM_EPOCHS
LOCAL_EPOCHS = 1

# Adjacency matrix
try:
    adj_path = f"datasets/{DATA_NAME}/adj_mx.pkl"
    _, adj_mx = load_adj(adj_path, "doubletransition")
    ADJ_MATRIX = torch.tensor(adj_mx, dtype=torch.float32)
except Exception:
    ADJ_MATRIX = None

# 베이스 모델 파라미터
BASE_MODEL_PARAMS = {
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
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
}

MODEL_PARAM = {
    "base_model_class": STAEformer,
    "base_model_params": BASE_MODEL_PARAMS,
    "num_clients": NUM_CLIENTS,
    "total_nodes": NUM_NODES,
    "output_dim": 1,
    "aggregator_type": AGGREGATOR_TYPE,
    "aggregator_params": AGGREGATOR_PARAMS,
    "partition_method": PARTITION_METHOD,
    "adj_matrix": ADJ_MATRIX,
    "client_nodes_list": None,
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'FederatedLearning + STAEformer on {DATA_NAME}'
CFG.GPU_NUM = 1
CFG.RUNNER = FederatedLearningRunner

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
CFG.MODEL.NAME = f'FedAvg_STAEformer'
CFG.MODEL.ARCH = FederatedLearningModel
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
CFG.TRAIN.NUM_EPOCHS = NUM_ROUNDS
CFG.TRAIN.NUM_ROUNDS = NUM_ROUNDS
CFG.TRAIN.LOCAL_EPOCHS = LOCAL_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'FedAvg_STAEformer',
    '_'.join([DATA_NAME, str(NUM_ROUNDS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0003}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": LR_MILESTONES, "gamma": 0.1}
# CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
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




