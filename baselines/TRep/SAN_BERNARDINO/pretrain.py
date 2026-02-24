"""
T-Rep Pre-training on SAN_BERNARDINO Dataset

Extends TS2Vec with learnable time embeddings and temporal relation prediction.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/TRep/SAN_BERNARDINO/pretrain.py', gpus='0')"
"""
import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.TRep.arch import TRepPretrainModel
from baselines.TRep.loss import get_trep_loss

############################## Hot Parameters ##############################
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture
NUM_NODES = 893
D_MODEL = 64
HIDDEN_DIM = 64
DEPTH = 10
TIME_EMBED_DIM = 64

# T-Rep loss weights
ALPHA = 0.5
TEMPORAL_UNIT = 0
INSTANCE_WEIGHT = 0.25
TEMPORAL_WEIGHT = 0.25
JSD_WEIGHT = 0.25
COND_WEIGHT = 0.25

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "input_dim": 3,  # flow, tod, dow
    "output_dim": 1,
    "d_model": D_MODEL,
    "hidden_dim": HIDDEN_DIM,
    "depth": DEPTH,
    "time_embed_dim": TIME_EMBED_DIM,
    "mask_mode": "binomial",
    "dropout": 0.1,
    "temporal_unit": TEMPORAL_UNIT,
}
NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'T-Rep Pre-training on SAN_BERNARDINO'
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
    'data_range': (0, 26280),  # 3 months only
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
CFG.MODEL.NAME = 'TRep_Pretrain'
CFG.MODEL.ARCH = TRepPretrainModel
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 3, 4]  # flow, tod, dow
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
    'TRep_pretrain_3mo',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = get_trep_loss(
    alpha=ALPHA,
    temporal_unit=TEMPORAL_UNIT,
    instance_weight=INSTANCE_WEIGHT,
    temporal_weight=TEMPORAL_WEIGHT,
    jsd_weight=JSD_WEIGHT,
    cond_weight=COND_WEIGHT,
)

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}

CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 8  # Reduced for memory (B*N similarity matrix)
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
