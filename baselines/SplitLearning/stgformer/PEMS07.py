import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SplitLearningRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.STGformer.arch.encoder import STGformerEncoder
from baselines.STGformer.arch.spatial import STGformerSpatial
from baselines.STGformer.arch.decoder import STGformerDecoder
from baselines.SplitLearning.arch.model import SplitLearningModel
from baselines.node_partition import setup_split_learning_nodes

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS07'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = SplitLearningModel

# Split Learning 설정
NUM_CLIENTS = 10
GROUPING_METHOD = 'random'

# 노드 분할 설정
NUM_NODES = 883
client_nodes_list, subgraph_adj_list, _ = setup_split_learning_nodes(
    num_nodes=NUM_NODES,
    num_clients=NUM_CLIENTS,
    grouping_method=GROUPING_METHOD,
    adj_matrix=None,
    dataset_name=DATA_NAME,
    load_adj_func=load_adj
)

# 공통 파라미터
INPUT_DIM = 3
OUTPUT_DIM = 1
INPUT_EMBEDDING_DIM = 24
TOD_EMBEDDING_DIM = 24
DOW_EMBEDDING_DIM = 24
SPATIAL_EMBEDDING_DIM = 0
ADAPTIVE_EMBEDDING_DIM = 24
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
DROPOUT_A = 0.1
KERNEL_SIZE = [1]
MLP_RATIO = 2
STEPS_PER_DAY = 288

# model_dim 계산
MODEL_DIM = (
    INPUT_EMBEDDING_DIM
    + TOD_EMBEDDING_DIM
    + DOW_EMBEDDING_DIM
    + SPATIAL_EMBEDDING_DIM
    + ADAPTIVE_EMBEDDING_DIM
)

# supports (full graph) 준비
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
supports = [torch.tensor(i) for i in adj_mx]

# Encoder 파라미터
ENCODER_PARAMS = {
    "in_steps": INPUT_LEN,
    "steps_per_day": STEPS_PER_DAY,
    "input_dim": INPUT_DIM,
    "input_embedding_dim": INPUT_EMBEDDING_DIM,
    "tod_embedding_dim": TOD_EMBEDDING_DIM,
    "dow_embedding_dim": DOW_EMBEDDING_DIM,
    "spatial_embedding_dim": SPATIAL_EMBEDDING_DIM,
    "adaptive_embedding_dim": ADAPTIVE_EMBEDDING_DIM,
    "dropout_a": DROPOUT_A,
    "kernel_size": KERNEL_SIZE,
    "supports": supports,
    "mlp_ratio": MLP_RATIO,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
}

# Decoder 파라미터
DECODER_PARAMS = {
    "model_dim": MODEL_DIM,
    "out_steps": OUTPUT_LEN,
    "output_dim": OUTPUT_DIM,
    "in_steps": INPUT_LEN,
    "kernel_size": KERNEL_SIZE,
    "mlp_ratio": MLP_RATIO,
    "dropout": DROPOUT,
    "num_layers": NUM_LAYERS,
}

# Server(STGformerSpatial) 파라미터
# Server는 전체 노드에 대한 spatial attention 수행
SERVER_SPATIAL_PARAMS = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "adaptive_embedding_dim": ADAPTIVE_EMBEDDING_DIM,
    "model_dim": MODEL_DIM,
    "mlp_ratio": MLP_RATIO,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
    "kernel_size": KERNEL_SIZE,
    "supports": supports,
    "order": 2,
}

# 통합 모델 파라미터 (SplitLearningModel)
MODEL_PARAM = {
    "num_clients": NUM_CLIENTS,
    "client_nodes_list": client_nodes_list,
    "total_nodes": NUM_NODES,
    "encoder_cls": STGformerEncoder,
    "decoder_cls": STGformerDecoder,
    "spatial_cls": STGformerSpatial,
    "encoder_params": ENCODER_PARAMS,
    "decoder_params": DECODER_PARAMS,
    "server_spatial_params": SERVER_SPATIAL_PARAMS,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "output_dim": OUTPUT_DIM,
}

NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STGformer Split Learning'
CFG.GPU_NUM = 1
CFG.RUNNER = SplitLearningRunner

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
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0003,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20, 25],
    "gamma": 0.1,
}
# CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20
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


