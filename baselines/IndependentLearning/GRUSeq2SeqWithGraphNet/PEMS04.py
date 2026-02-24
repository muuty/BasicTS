#!/usr/bin/env python3
"""
Independent Learning with GRUSeq2SeqWithGraphNet on PEMS04

각 클라이언트가 자신의 노드 서브셋에 대해 독립적인 모델을 학습합니다.
"""

import os
import sys
import torch
import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import IndependentLearningRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.GRUSeq2SeqWithGraphNet.arch import GRUSeq2SeqWithGraphNet
from baselines.IndependentLearning.arch import IndependentLearningModel
from baselines.node_partition import setup_split_learning_nodes


def adj_to_edge_index(adj_mx):
    """Convert adjacency matrix to edge_index and edge_attr format."""
    if isinstance(adj_mx, np.ndarray):
        adj_mx = torch.from_numpy(adj_mx).float()
    
    edge_index = (adj_mx > 0).nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj_mx[adj_mx > 0]
    
    return edge_index, edge_attr


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS04'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Independent Learning 설정
NUM_CLIENTS = 10
TOTAL_NODES = 307
PARTITION_METHOD = 'random'  # 'random', 'metis', 'spectral'

# 인접 행렬 로드 (graph-based partitioning용)
try:
    adj_mx, _ = load_adj(f'datasets/{DATA_NAME}/adj_mx.pkl', 'doubletransition')
    adj_mx_np = adj_mx[0] if isinstance(adj_mx, list) else adj_mx
    edge_index, edge_attr = adj_to_edge_index(adj_mx_np)
    ADJ_MATRIX = torch.tensor(adj_mx_np, dtype=torch.float32)
except Exception:
    ADJ_MATRIX = None
    edge_index = None
    edge_attr = None
    print(f"Warning: Could not load adjacency matrix for {DATA_NAME}")

# 노드 분할 및 subgraph_adj_list 생성
client_nodes_list, subgraph_adj_list, _ = setup_split_learning_nodes(
    num_nodes=TOTAL_NODES,
    num_clients=NUM_CLIENTS,
    grouping_method=PARTITION_METHOD,
    adj_matrix=ADJ_MATRIX,
    dataset_name=DATA_NAME,
    load_adj_func=load_adj
)

# GRUSeq2SeqWithGraphNet 베이스 모델 파라미터 (num_nodes, edge_index, edge_attr 제외 - wrapper에서 설정)
BASE_MODEL_PARAMS = {
    "input_dim": 1,  # flow only
    "output_dim": 1,
    "hidden_size": 128,
    "gru_num_layers": 2,
    "dropout": 0.0,
    "cl_decay_steps": 2000,
    "use_curriculum_learning": True,
    "gn_layer_num": 2,
    "gn_hidden_size": 256,
    "gn_updated_node_size": 128,
    "gn_updated_edge_size": 128,
    "gn_updated_global_size": 128,
    "seq_len": INPUT_LEN,
    "horizon": OUTPUT_LEN,
    "edge_index": None,  # 클라이언트별로 설정됨
    "edge_attr": None,  # 클라이언트별로 설정됨
}

# IndependentLearningModel 파라미터
MODEL_ARCH = IndependentLearningModel
MODEL_PARAM = {
    "base_model_class": GRUSeq2SeqWithGraphNet,
    "base_model_params": BASE_MODEL_PARAMS,
    "num_clients": NUM_CLIENTS,
    "total_nodes": TOTAL_NODES,
    "partition_method": PARTITION_METHOD,
    "adj_matrix": ADJ_MATRIX,
    "output_dim": 1,
    "client_nodes_list": client_nodes_list,  # setup_split_learning_nodes에서 생성
    "subgraph_adj_list": None,  # GRUSeq2SeqWithGraphNet은 subgraph_adj_list 불필요
}

NUM_EPOCHS = 300

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'Independent Learning with GRUSeq2SeqWithGraphNet ({NUM_CLIENTS} clients)'
CFG.GPU_NUM = 1

# IndependentLearningRunner 사용
CFG.RUNNER = IndependentLearningRunner

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
CFG.MODEL.NAME = f'IndependentLearning_{GRUSeq2SeqWithGraphNet.__name__}_{NUM_CLIENTS}clients'
CFG.MODEL.ARCH = MODEL_ARCH
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
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    f'IndependentLearning_{GRUSeq2SeqWithGraphNet.__name__}',
    '_'.join([DATA_NAME, str(NUM_CLIENTS), str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0003
}

# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50, 80],
    "gamma": 0.1
}

# Early stopping
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20

# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
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

