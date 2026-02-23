import os
import sys
import torch
import random
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import DGCRN
from .runner import DGCRNRunner

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'xtraffic/SAN_BERNARDINO'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = DGCRN
adj_mx, _ = load_adj("datasets/" + DATA_NAME +
                     "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "gcn_depth": 2,
    "num_nodes": 893,
    "predefined_A": [torch.Tensor(_) for _ in adj_mx],
    "dropout": 0.3,
    "subgraph_size": 20,
    "node_dim": 40,
    "middle_dim": 2,
    "seq_length": 12,
    "in_dim": 2,
    "list_weight": [0.05, 0.95, 0.95],
    "tanhalpha": 3,
    "cl_decay_steps": 4000,
    "rnn_size": 64,
    "hyperGNN_dim": 16,
    "use_gate": True,
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = DGCRNRunner
# DCRNN does not allow to load parameters since it creates parameters in the first iteration
CFG._ = random.randint(int(-1e6), int(1e6))

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
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
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
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

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
    '_'.join([DATA_NAME, "REPLAY_FULL_UPDATE_001", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr":0.001,
    "weight_decay":0.0001
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones":[100, 150],
    "gamma": 0.5
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True
# Gradient clipping settings
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
# Curriculum learning
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = 12

# Early stopping
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 20 

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

############################## Experience Replay Configuration ##############################

CFG.EXPERIENCE_REPLAY = EasyDict()
# Experience replay buffer settings
CFG.EXPERIENCE_REPLAY.CAPACITY = 1024 # Maximum number of samples in replay buffer
CFG.EXPERIENCE_REPLAY.TOPK_SAMPLES = 2  # Number of top-K samples to enqueue per batch (based on MAE)
CFG.EXPERIENCE_REPLAY.TOPQ_NODES = 0.10  # Fraction of top nodes to mask (e.g., 0.10 = top 10% nodes)
CFG.EXPERIENCE_REPLAY.NODE_THR = None  # Threshold for node masking (None = use TOPQ_NODES, float = absolute threshold)
CFG.EXPERIENCE_REPLAY.SEED = None  # Random seed for replay buffer (None = no fixed seed)
CFG.EXPERIENCE_REPLAY.BATCH_SIZE = 32  # Batch size for replay updates
CFG.EXPERIENCE_REPLAY.UPDATE_TARGET = 'all'  # Update target: 'gate' (only gate parameters) or 'all' (all model parameters)
CFG.EXPERIENCE_REPLAY.LR = 0.001  # Learning rate for replay optimizer


############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
