import os
import sys
import torch
import numpy as np
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import GRUSeq2SeqGraphNetHierModel
from baselines.node_partition import setup_split_learning_nodes


def adj_to_edge_index(adj_mx, node_indices=None):
    """
    Convert adjacency matrix to edge_index and edge_weight format.
    
    Args:
        adj_mx: Full adjacency matrix
        node_indices: If provided, extract subgraph for these nodes
    """
    if isinstance(adj_mx, np.ndarray):
        adj_mx = torch.from_numpy(adj_mx).float()
    
    if node_indices is not None:
        # Extract subgraph
        node_indices = list(node_indices)
        sub_adj = adj_mx[node_indices][:, node_indices]
    else:
        sub_adj = adj_mx
    
    edge_index = (sub_adj > 0).nonzero(as_tuple=False).t().contiguous()
    edge_weight = sub_adj[sub_adj > 0]
    
    return edge_index, edge_weight


############################## Hot Parameters ##############################
DATA_NAME = 'PEMS04'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = GRUSeq2SeqGraphNetHierModel

# Hierarchical Split Learning 설정
NUM_CLIENTS = 10
NUM_TOKENS = 1
POOLING_METHOD = 'attention'
GROUPING_METHOD = 'random'

# 노드 분할 설정
NUM_NODES = 307
client_nodes_list, subgraph_adj_list, _ = setup_split_learning_nodes(
    num_nodes=NUM_NODES,
    num_clients=NUM_CLIENTS,
    grouping_method=GROUPING_METHOD,
    adj_matrix=None,
    dataset_name=DATA_NAME,
    load_adj_func=load_adj
)

# Load adjacency matrix for GraphNet
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
adj_mx_np = adj_mx[0] if isinstance(adj_mx, list) else adj_mx

# Create edge_index and edge_weight for each client's subgraph
edge_index_list = []
edge_weight_list = []
for client_nodes in client_nodes_list:
    edge_index, edge_weight = adj_to_edge_index(adj_mx_np, client_nodes)
    edge_index_list.append(edge_index)
    edge_weight_list.append(edge_weight)

# 클라이언트 모델 파라미터
CLIENT_MODEL_PARAMS = {
    "input_dim": 1,
    "hidden_size": 128,
    "output_dim": 1,
    "gru_num_layers": 2,
    "dropout": 0.0,
    "cl_decay_steps": 1000,
    "use_curriculum_learning": True,
    "seq_len": INPUT_LEN,
    "horizon": OUTPUT_LEN,
    "gn_num_blocks": 2,
}

# 서버 모델 파라미터
SERVER_MODEL_PARAMS = {
    "num_nodes": NUM_NODES,
    "hidden_dim": 128,
    "output_dim": 1,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "feed_forward_dim": 256,
    "horizon": OUTPUT_LEN,
}

# 통합 모델 파라미터
MODEL_PARAM = {
    "num_clients": NUM_CLIENTS,
    "client_nodes_list": client_nodes_list,
    "client_model_params": CLIENT_MODEL_PARAMS,
    "server_model_params": SERVER_MODEL_PARAMS,
    "num_tokens": NUM_TOKENS,
    "pooling_method": POOLING_METHOD,
    "subgraph_adj_list": subgraph_adj_list,
    "edge_index_list": edge_index_list,
    "edge_weight_list": edge_weight_list,
}

NUM_EPOCHS = 300

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'GRU Seq2Seq + GraphNet Hierarchical Split Learning'
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
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0003
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50, 80],
    "gamma": 0.1
}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

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

