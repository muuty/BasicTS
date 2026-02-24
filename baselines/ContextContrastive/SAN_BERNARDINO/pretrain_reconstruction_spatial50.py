"""
Reconstruction-Only Pre-training (MAE-style)

No contrastive loss - only reconstruction loss.
Masks 50% of nodes and reconstructs them.

Key benefits over contrastive:
- No false negative problem
- Direct signal for learning spatial relationships
- Simpler training objective

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_reconstruction_spatial50.py', gpus='0')"
"""
import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.ContextContrastive.arch import UnifiedPretrainModel
from baselines.ContextContrastive.runner import PretrainRunner

############################## Hot Parameters ##############################
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

NUM_NODES = 893
D_MODEL = 64
NUM_LAYERS = 2
NHEAD = 4
NUM_EPOCHS = 30

############################## MODEL CONFIGURATION ##############################
MODEL_PARAM = {
    'adj_path': f'datasets/{DATA_NAME}/adj_mx.pkl',
    'encoder': {
        'type': 'TransformerEncoder',
        'input_dim': 3,
        'd_model': D_MODEL,
        'num_layers': NUM_LAYERS,
        'nhead': NHEAD,
        'dropout': 0.1,
    },
    # Masking: spatial only (50%)
    'masking': {
        'spatial': {'enabled': True, 'ratio': 0.50},
        'feature': {'enabled': False, 'ratio': 0.30},
        'temporal': {'enabled': False, 'ratio': 0.15},
    },
    # Heads: reconstruction only (no contrastive)
    'heads': {
        'contrastive': False,      # DISABLED - no negative pairs!
        'reconstruction': True,    # ENABLED - MAE-style
    },
    'proj_dim': D_MODEL,
}

############################## PRETRAIN CONFIGURATION ##############################
PRETRAIN_CONFIG = {
    'LOSS': {
        'contrastive': {
            'enabled': False,  # DISABLED
        },
        'reconstruction': {
            'enabled': True,   # ENABLED
            'weight': 1.0,
            'target': 'masked',  # Only reconstruct masked positions
            'loss_type': 'mse',
        },
    },
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Reconstruction-Only Pretrain: Spatial Masking 50% (No False Negatives)'
CFG.GPU_NUM = 1
CFG.RUNNER = PretrainRunner
CFG.PRETRAIN = PRETRAIN_CONFIG

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    'data_range': (0, 26280),  # 3 months
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
CFG.MODEL.NAME = 'ContextContrastive_reconstruction_spatial50_3mo'
CFG.MODEL.ARCH = UnifiedPretrainModel
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
    '_'.join([DATA_NAME.replace('/', '_'), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = None  # Handled by PretrainRunner

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}

CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
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
