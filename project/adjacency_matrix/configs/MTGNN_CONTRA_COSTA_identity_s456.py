import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.MTGNN.arch import MTGNN
from baselines.MTGNN.runner import MTGNNRunner
from project.adjacency_matrix import get_adjacency, normalize_for_model

############################## Experiment Parameters ##############################
DATA_NAME = 'CONTRA_COSTA'
ADJ_METHOD = 'identity'
SEED = 456

regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Adjacency injection (from sensor coordinates)
raw_adj = get_adjacency(ADJ_METHOD, DATA_NAME)
predefined_A = normalize_for_model(raw_adj, 'MTGNN')

num_nodes = 773
MODEL_ARCH = MTGNN
MODEL_PARAM = {
    "gcn_true": True,
    "buildA_true": False,
    "gcn_depth": 2,
    "num_nodes": num_nodes,
    "predefined_A": predefined_A,
    "dropout": 0.3,
    "subgraph_size": 20,
    "node_dim": 40,
    "dilation_exponential": 1,
    "conv_channels": 32,
    "residual_channels": 32,
    "skip_channels": 64,
    "end_channels": 128,
    "seq_length": 12,
    "in_dim": 2,
    "out_dim": 12,
    "layers": 3,
    "propalpha": 0.05,
    "tanhalpha": 3,
    "layer_norm_affline": True,
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'MTGNN_{DATA_NAME}_{ADJ_METHOD}_s{SEED}'
CFG.GPU_NUM = 1
CFG.RUNNER = MTGNNRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = SEED
CFG.ENV.CUDNN_ENABLED = True

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    'data_range': (0, 24192),
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
CFG.MODEL.FORWARD_FEATURES = [0, 3]  # flow + tod (xtraffic channel order)
CFG.MODEL.TARGET_FEATURES = [0]       # flow

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
    'checkpoints', 'adj_experiment', 'MTGNN',
    f'{DATA_NAME}_{ADJ_METHOD}_s{SEED}'
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 3
CFG.TRAIN.CL.PREDICTION_LENGTH = 12
CFG.TRAIN.CUSTOM = EasyDict()
CFG.TRAIN.CUSTOM.STEP_SIZE = 100
CFG.TRAIN.CUSTOM.NUM_NODES = num_nodes
CFG.TRAIN.CUSTOM.NUM_SPLIT = 1
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 15

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
