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

from baselines.STAEformer.arch.encoder import STAEformerEncoder
from baselines.STAEformer.arch.spatial import STAEformerSpatial
from baselines.STAEformer.arch.decoder import STAEformerDecoder
from baselines.HierSplit.arch.model import HierSplitModel
from baselines.node_partition import load_metadata_csv
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

ADJ_MATRIX, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
ADJ_MATRIX = torch.Tensor(ADJ_MATRIX[0])

MODEL_ARCH = HierSplitModel

# Hierarchical Split Learning 설정
NUM_NODES = 307
NUM_CLIENTS = 10
NUM_TOKENS = max(1, int(round(NUM_NODES / NUM_CLIENTS / 5)))  # 논문 취지(≈20%) 반영
POOLING_METHOD = 'attention'
PARTITION_METHOD = 'random'
METADATA = None



# 공통 파라미터
INPUT_DIM = 3
OUTPUT_DIM = 1
INPUT_EMBEDDING_DIM = 24
TOD_EMBEDDING_DIM = 24
DOW_EMBEDDING_DIM = 24
SPATIAL_EMBEDDING_DIM = 0
ADAPTIVE_EMBEDDING_DIM = 24
FEED_FORWARD_DIM = 256
NUM_HEADS = 4
DROPOUT = 0.1
STEPS_PER_DAY = 288

# model_dim 계산
MODEL_DIM = (
    INPUT_EMBEDDING_DIM
    + TOD_EMBEDDING_DIM
    + DOW_EMBEDDING_DIM
    + SPATIAL_EMBEDDING_DIM
    + ADAPTIVE_EMBEDDING_DIM
)

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
    "feed_forward_dim": FEED_FORWARD_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": 1,
    "dropout": DROPOUT,
}

# Client Spatial 파라미터 (Local Spatial Attention)
SPATIAL_PARAMS = {
    "model_dim": MODEL_DIM,
    "feed_forward_dim": FEED_FORWARD_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": 1,
    "dropout": DROPOUT,
}

# Server Spatial 파라미터 (Global Spatial Attention on Tokens)
SERVER_SPATIAL_PARAMS = {
    "model_dim": MODEL_DIM,
    "feed_forward_dim": FEED_FORWARD_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": 1,  # Server는 보통 1 layer
    "dropout": DROPOUT,
}

# Decoder 파라미터
DECODER_PARAMS = {
    "model_dim": MODEL_DIM,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "output_dim": OUTPUT_DIM,
    "use_mixed_proj": True,     
}

# 통합 모델 파라미터
MODEL_PARAM = {
    # Client 구성
    "num_clients": NUM_CLIENTS,
    "total_nodes": NUM_NODES,
    "encoder_cls": STAEformerEncoder,
    "spatial_cls": STAEformerSpatial,
    "decoder_cls": STAEformerDecoder,
    "encoder_params": ENCODER_PARAMS,
    "spatial_params": SPATIAL_PARAMS,
    "decoder_params": DECODER_PARAMS,
    # Server 구성
    "server_spatial_params": SERVER_SPATIAL_PARAMS,
    # Pooling 설정
    "model_dim": MODEL_DIM,
    "num_tokens": NUM_TOKENS,
    "pooling_method": POOLING_METHOD,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
    "partition_method": PARTITION_METHOD,
    "adj_matrix": ADJ_MATRIX,
    "metadata": METADATA,
    # Output 설정
    "out_steps": OUTPUT_LEN,
    "output_dim": OUTPUT_DIM,
}

NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STAEformer Hierarchical Split Learning'
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
    "gamma": 0.1
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

