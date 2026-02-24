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

from baselines.STGCN.arch.decoder import STGCNDecoder
from baselines.STGCN.arch.encoder import STGCNEncoder
from baselines.STGCN.arch.spatial import STGCNServerSpatial
from baselines.HierSplit.arch.model import HierSplitModel
from baselines.node_partition import setup_split_learning_nodes

############################## Hot Parameters ##############################
DATA_NAME = 'PEMS-BAY'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = HierSplitModel

# Hierarchical Split Learning 설정
NUM_CLIENTS = 10
NUM_TOKENS = 1
POOLING_METHOD = 'attention'
GROUPING_METHOD = 'random'

# 노드 분할 설정
NUM_NODES = 325
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
adj_mx = torch.Tensor(adj_mx[0])

Kt = 3
Ks = 3
T = 12  # input timesteps
OUTPUT_LEN = 12

# STGCN 원래 blocks 구조
# blocks = [[input_dim], [c1_in, c1_mid, c1_out], [c2_in, c2_mid, c2_out], [fc1, fc2], [output]]
blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]]

# 시간 차원 계산
# 각 STConvBlock은 2*(Kt-1) 만큼 시간 차원 감소
num_st_blocks = 2  # STConvBlock 개수
T_after_encoder = T - num_st_blocks * 2 * (Kt - 1)  # 12 - 2*2*2 = 4

# HierSplit용 model_dim (Encoder 출력 채널)
MODEL_DIM = blocks[2][-1]  # 64 (두 번째 STConvBlock 출력)

# ─────────────────────────────────────────────────────────────
# Encoder: 2개의 STConvBlock 전체
# ─────────────────────────────────────────────────────────────
ENCODER_PARAMS = {
    "Kt": Kt,
    "Ks": Ks,
    "last_block_channel": blocks[0][0],  # 1 (input channel)
    "channels": blocks[1],  # [64, 16, 64] - 첫 번째 STConvBlock
    "channels_2": blocks[2],  # [64, 16, 64] - 두 번째 STConvBlock (추가)
    "act_func": "glu",
    "graph_conv_type": "cheb_graph_conv",
    "bias": True,
    "droprate": 0.5,
    "adj_matrix": adj_mx,
    # num_nodes, gso는 동적으로 설정됨
}

# ─────────────────────────────────────────────────────────────
# Spatial: Identity (Encoder에 이미 포함)
# ─────────────────────────────────────────────────────────────
SPATIAL_PARAMS = {}

# ─────────────────────────────────────────────────────────────
# Server: Global Attention on Tokens
# ─────────────────────────────────────────────────────────────
SERVER_SPATIAL_PARAMS = {
    "model_dim": MODEL_DIM,  # 64
    "num_heads": 4,
    "dropout": 0.5,
}

# ─────────────────────────────────────────────────────────────
# Decoder: OutputBlock만
# ─────────────────────────────────────────────────────────────
DECODER_PARAMS = {
    "Ko": T_after_encoder,  # 4 (Encoder 통과 후 남은 시간 차원)
    "last_block_channel": blocks[2][-1],  # 64 (Encoder 출력 채널)
    "output_channels": blocks[-2],  # [128, 128]
    "out_steps": OUTPUT_LEN,  # 12
    "act_func": "glu",
    "bias": True,
    "droprate": 0.5,
    # num_nodes는 동적으로 설정됨
}

# ─────────────────────────────────────────────────────────────
# 전체 모델 파라미터
# ─────────────────────────────────────────────────────────────
MODEL_PARAM = {
    # Client 구성
    "num_clients": NUM_CLIENTS,
    "total_nodes": NUM_NODES,
    "encoder_cls": STGCNEncoder,
    "spatial_cls": STGCNServerSpatial,
    "decoder_cls": STGCNDecoder,
    "encoder_params": ENCODER_PARAMS,
    "spatial_params": SPATIAL_PARAMS,
    "decoder_params": DECODER_PARAMS,
    
    # Server 구성
    "server_spatial_params": SERVER_SPATIAL_PARAMS,
    
    # Pooling 설정
    "model_dim": MODEL_DIM,  # 64
    "num_tokens": NUM_TOKENS,
    "pooling_method": POOLING_METHOD,
    "num_heads": 4,
    "dropout": 0.5,
    
    # Partition 설정
    "partition_method": GROUPING_METHOD,
    "adj_matrix": adj_mx,
    "metadata": None,
    
    # Output 설정
    "out_steps": OUTPUT_LEN,
    "output_dim": 1,
}


NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STGCN Hierarchical Split Learning'
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
    "gamma": 0.5
}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20
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




































