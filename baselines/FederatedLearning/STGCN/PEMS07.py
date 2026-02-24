#!/usr/bin/env python3
"""
Federated Learning + STGCN Configuration for PEMS07 Dataset

Usage:
    python experiments/train.py -c baselines/FederatedLearning/STGCN/STGCN-PEMS07.py -g 0
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
from ...STGCN.arch import STGCN

############################## Hot Parameters ##############################
DATA_NAME = 'PEMS07'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Dataset specific
NUM_NODES = 883
NUM_EPOCHS = 30
LR_MILESTONES = [1, 50]
BATCH_SIZE = 16

# Federated Learning 설정
NUM_CLIENTS = 4
PARTITION_METHOD = 'random'
AGGREGATOR_TYPE = 'fedavg'
AGGREGATOR_PARAMS = {}
NUM_ROUNDS = NUM_EPOCHS
LOCAL_EPOCHS = 1

# Adjacency matrix (STGCN는 gso로 직접 사용)
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
adj_mx = torch.Tensor(adj_mx[0])

# 베이스 모델 파라미터 (STGCN)
BASE_MODEL_PARAMS = {
    "Ks": 3,
    "Kt": 3,
    "blocks": [[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
    "T": 12,
    "n_vertex": NUM_NODES,
    "act_func": "glu",
    "graph_conv_type": "cheb_graph_conv",
    "gso": adj_mx,
    "bias": True,
    "droprate": 0.5,
}

MODEL_PARAM = {
    "base_model_class": STGCN,
    "base_model_params": BASE_MODEL_PARAMS,
    "num_clients": NUM_CLIENTS,
    "total_nodes": NUM_NODES,
    "output_dim": 1,
    "aggregator_type": AGGREGATOR_TYPE,
    "aggregator_params": AGGREGATOR_PARAMS,
    "partition_method": PARTITION_METHOD,
    "adj_matrix": adj_mx,
    "client_nodes_list": None,
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'FederatedLearning + STGCN on {DATA_NAME}'
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
CFG.MODEL.NAME = f'FedAvg_STGCN'
CFG.MODEL.ARCH = FederatedLearningModel
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
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
    'FedAvg_STGCN',
    '_'.join([DATA_NAME, str(NUM_ROUNDS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.0004, "weight_decay": 0.0003}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": LR_MILESTONES, "gamma": 0.5}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20
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
