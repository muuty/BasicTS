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

from baselines.STGCN.arch.encoder import STGCNEncoder
from baselines.STGCN.arch.spatial import STGCNServerSpatial
from baselines.STGCN.arch.decoder import STGCNDecoder
from baselines.SplitLearning.arch.model import SplitLearningModel

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS03'
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

# 노드 설정
NUM_NODES = 358

# STGCN 파라미터
KS = 3
KT = 3
BLOCKS = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]]
ACT_FUNC = 'glu'
GRAPH_CONV_TYPE = 'cheb_graph_conv'
BIAS = True
DROPRATE = 0.5

# adjacency matrix 로드
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
adj_mx = torch.Tensor(adj_mx[0])

# Ko 계산: T - 2 * (Kt - 1) * 2 = 12 - 4 * 2 = 4
KO = INPUT_LEN - 2 * (KT - 1) * 2

# Encoder 파라미터
ENCODER_PARAMS = {
    "Kt": KT,
    "Ks": KS,
    "last_block_channel": BLOCKS[0][0],
    "channels": BLOCKS[1],
    "channels_2": BLOCKS[2],
    "act_func": ACT_FUNC,
    "graph_conv_type": GRAPH_CONV_TYPE,
    "adj_matrix": adj_mx,
    "bias": BIAS,
    "droprate": DROPRATE,
}

# Decoder 파라미터
DECODER_PARAMS = {
    "Ko": KO,
    "last_block_channel": BLOCKS[2][-1],
    "output_channels": BLOCKS[3],
    "out_steps": OUTPUT_LEN,
    "act_func": ACT_FUNC,
    "bias": BIAS,
    "droprate": DROPRATE,
}

# Server Spatial 파라미터 (학습 가능한 GraphConvLayer 기반)
SERVER_SPATIAL_PARAMS = {
    "graph_conv_type": GRAPH_CONV_TYPE,
    "feature_dim": BLOCKS[2][-1],
    "Ks": KS,
    "adj_matrix": adj_mx,
    "bias": BIAS,
    "droprate": DROPRATE,
}

# 통합 모델 파라미터
MODEL_PARAM = {
    "num_clients": NUM_CLIENTS,
    "total_nodes": NUM_NODES,
    "encoder_cls": STGCNEncoder,
    "decoder_cls": STGCNDecoder,
    "spatial_cls": STGCNServerSpatial,
    "encoder_params": ENCODER_PARAMS,
    "decoder_params": DECODER_PARAMS,
    "server_spatial_params": SERVER_SPATIAL_PARAMS,
    "partition_method": GROUPING_METHOD,
    "adj_matrix": adj_mx,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "output_dim": 1,
}

NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STGCN Split Learning'
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
    "lr": 0.0004,
    "weight_decay": 0.0003,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50],
    "gamma": 0.5,
}
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

