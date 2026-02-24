"""
Spatial + Feature Masking Pre-training

Combines the best performing augmentation (spatial/node masking) with feature masking.

Hypothesis:
    - Node masking: learns spatial relationships between nodes (PROVEN BEST)
    - Feature masking: learns cross-modal relationships (flow <-> tod/dow)
    - Combined: spatial context + temporal pattern awareness

Key difference from combined_v1/v2/v3:
    Those used feature + temporal masking (NO spatial).
    This uses node (spatial) + feature masking.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_spatial_feature.py', gpus='0')"
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

from ..arch import ConfigurablePretrainModel
from ..loss import get_contrastive_loss

############################## Hot Parameters ##############################
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = ConfigurablePretrainModel
NUM_NODES = 893
D_MODEL = 64
NUM_LAYERS = 2
NHEAD = 4
TEMPERATURE = 0.1

# Augmentation: SPATIAL (Node) + FEATURE Masking
# Node masking 15%: proven best for spatial relationship learning
# Feature masking 15%: learn cross-modal (flow <-> tod/dow) relationships
AUGMENTATION_CONFIG = {
    'node_masking': {'mask_ratio': 0.15},
    'feature_masking': {'mask_ratio': 0.15},
}

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "input_dim": 3,  # flow, tod, dow
    "output_dim": 1,
    "d_model": D_MODEL,
    "num_layers": NUM_LAYERS,
    "nhead": NHEAD,
    "dropout": 0.1,
    "augmentation_config": AUGMENTATION_CONFIG,
}
NUM_EPOCHS = 30  # Same as spatial masking

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Contrastive Pre-training: Spatial + Feature Masking'
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
CFG.MODEL.NAME = 'ContextContrastive_spatial_feature_3mo'
CFG.MODEL.ARCH = MODEL_ARCH
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
    CFG.MODEL.NAME,
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.DATASET.PARAM.input_len), str(CFG.DATASET.PARAM.output_len)])
)
CFG.TRAIN.LOSS = get_contrastive_loss(temperature=TEMPERATURE)
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 1e-5}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16  # Reduced from 32 to avoid OOM in contrastive loss
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16  # Reduced to match train batch size

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16  # Reduced to avoid OOM
