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

from .arch import GRUSeq2SeqWithGraphNet


def adj_to_edge_index(adj_mx):
    """Convert adjacency matrix to edge_index and edge_attr format.
    
    Args:
        adj_mx: Adjacency matrix [N, N]
        
    Returns:
        edge_index: [2, E] tensor
        edge_attr: [E] tensor
    """
    if isinstance(adj_mx, np.ndarray):
        adj_mx = torch.from_numpy(adj_mx).float()
    
    # Get non-zero edges
    edge_index = (adj_mx > 0).nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj_mx[adj_mx > 0]
    
    return edge_index, edge_attr


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'METR-LA'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']  # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE']  # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL']  # Null value in the data

# Model architecture and parameters
MODEL_ARCH = GRUSeq2SeqWithGraphNet

# Load adjacency matrix and convert to edge format
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
# Use the first adjacency matrix (forward transition)
adj_mx_np = adj_mx[0] if isinstance(adj_mx, list) else adj_mx
edge_index, edge_attr = adj_to_edge_index(adj_mx_np)

MODEL_PARAM = {
    "num_nodes": 207,
    "input_dim": 1,  # features to use (flow only)
    "output_dim": 1,
    "hidden_size": 128,
    "gru_num_layers": 2,
    "dropout": 0.0,
    "cl_decay_steps": 1000,
    "use_curriculum_learning": True,
    "gn_layer_num": 2,
    "gn_hidden_size": 256,
    "gn_updated_node_size": 128,
    "gn_updated_edge_size": 128,
    "gn_updated_global_size": 128,
    "seq_len": INPUT_LEN,
    "horizon": OUTPUT_LEN,
    "edge_index": edge_index,
    "edge_attr": edge_attr,
}
NUM_EPOCHS = 300

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'GRUSeq2SeqWithGraphNet for METR-LA'
CFG.GPU_NUM = 1  # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler  # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]  # flow only
CFG.MODEL.TARGET_FEATURES = [0]  # flow

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
# Metrics settings
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
    "eps": 1e-3
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
# Gradient clipping settings
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

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12]  # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True  # Whether to use GPU for evaluation. Default: True

