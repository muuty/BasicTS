"""
Stage 2: STAEformer with Pre-trained Encoder for PEMS08

Uses the pre-trained TemporalEncoder to transform input data,
then trains STAEformer on the encoded representations.

Prerequisites:
    Run Stage 1 first: baselines/ContextContrastive/PEMS08/pretrain.py

Usage:
    python -m basicts.run --cfg baselines/ContextContrastive/PEMS08/STAEformer.py --gpus 0
"""
import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.STAEformer.arch import STAEformer
from ..runner import PretrainedEncoderRunner

############################## Hot Parameters ##############################
DATA_NAME = 'PEMS08'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Pre-trained encoder config
D_MODEL = 64

# STAEformer model
MODEL_ARCH = STAEformer
ADJ_MATRIX, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
ADJ_MATRIX = torch.Tensor(ADJ_MATRIX[0])

MODEL_PARAM = {
    "num_nodes": 170,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 288,
    "input_dim": D_MODEL + 2,  # D_MODEL + 2 for [placeholder, tod, dow, encoded_rest] structure
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,   # Enabled - same as baseline for fair comparison
    "dow_embedding_dim": 24,   # Enabled - same as baseline for fair comparison
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
    "adj_matrix": ADJ_MATRIX,
}
NUM_EPOCHS = 30

############################## Pre-trained Encoder ##############################
# UPDATE THIS PATH after running Stage 1!
PRETRAINED_CKPT = 'checkpoints/ContextContrastive_pretrain/PEMS08_50_12_12/4e08bf9303d45ae6dc4c084d44776987/ContextContrastiveModel_best_val_MAE.pt'

PRETRAINED_ENCODER = {
    'ckpt_path': PRETRAINED_CKPT,
    'input_dim': 3,
    'd_model': D_MODEL,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,
    'include_tod_dow': True,  # Include tod/dow for STAEformer's temporal embeddings
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Stage 2: STAEformer with Pre-trained Encoder'
CFG.GPU_NUM = 1
CFG.RUNNER = PretrainedEncoderRunner
CFG.PRETRAINED_ENCODER = PRETRAINED_ENCODER

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
CFG.MODEL.NAME = 'STAEformer_Pretrained'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]  # All features go through encoder
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
    'ContextContrastive_STAEformer',
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0003}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.1}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
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
