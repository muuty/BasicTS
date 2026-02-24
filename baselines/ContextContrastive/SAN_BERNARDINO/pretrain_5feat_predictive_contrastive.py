"""Pretrain with Predictive Contrastive Loss (5feat)

Single TransformerEncoder trained to predict future representations from past.
Loss: cosine similarity with stop-gradient (SimSiam-style).

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_5feat_predictive_contrastive.py', gpus='1')"
"""
import os, sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.ContextContrastive.arch import PredictiveContrastiveModel
from baselines.ContextContrastive.runner import PredictiveContrastiveRunner

DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN, OUTPUT_LEN = regular_settings['INPUT_LEN'], regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
RESCALE, NULL_VAL = regular_settings['RESCALE'], regular_settings['NULL_VAL']

NUM_NODES, D_MODEL = 893, 64
NUM_EPOCHS = 30

MODEL_PARAM = {
    'input_dim': 5,
    'd_model': D_MODEL,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,
    'dim_feedforward': 256,
    'predictor_hidden': 128,
}

CFG = EasyDict()
CFG.DESCRIPTION = 'Predictive Contrastive Pretrain (5feat, d=64)'
CFG.GPU_NUM = 1
CFG.RUNNER = PredictiveContrastiveRunner

CFG.DATASET = EasyDict({
    'NAME': DATA_NAME,
    'TYPE': TimeSeriesForecastingDataset,
    'PARAM': EasyDict({
        'dataset_name': DATA_NAME,
        'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
        'data_range': (0, 26280),
    })
})

CFG.SCALER = EasyDict({
    'TYPE': ZScoreScaler,
    'PARAM': EasyDict({
        'dataset_name': DATA_NAME,
        'train_ratio': TRAIN_VAL_TEST_RATIO[0],
        'norm_each_channel': True,
        'rescale': RESCALE,
        'target_channel': [0, 1, 2],
    })
})

CFG.MODEL = EasyDict({
    'NAME': 'PredictiveContrastive_5feat_3mo',
    'ARCH': PredictiveContrastiveModel,
    'PARAM': MODEL_PARAM,
    'FORWARD_FEATURES': [0, 1, 2, 3, 4],
    'TARGET_FEATURES': [0],
})

CFG.METRICS = EasyDict({
    'FUNCS': EasyDict({
        'MAE': masked_mae,
        'MAPE': masked_mape,
        'RMSE': masked_rmse,
    }),
    'TARGET': 'MAE',
    'NULL_VAL': NULL_VAL,
})

CFG.TRAIN = EasyDict({
    'NUM_EPOCHS': NUM_EPOCHS,
    'CKPT_SAVE_DIR': os.path.join(
        'checkpoints',
        CFG.MODEL.NAME,
        '_'.join([DATA_NAME.replace('/', '_'), str(INPUT_LEN), str(OUTPUT_LEN)])
    ),
    'LOSS': None,
    'OPTIM': EasyDict({
        'TYPE': 'Adam',
        'PARAM': {'lr': 0.001, 'weight_decay': 0.0001},
    }),
    'LR_SCHEDULER': EasyDict({
        'TYPE': 'CosineAnnealingLR',
        'PARAM': {'T_max': NUM_EPOCHS, 'eta_min': 1e-6},
    }),
    'CLIP_GRAD_PARAM': {'max_norm': 5.0},
    'DATA': EasyDict({'BATCH_SIZE': 16, 'SHUFFLE': True}),
})

CFG.VAL = EasyDict({
    'INTERVAL': 1,
    'DATA': EasyDict({'BATCH_SIZE': 32}),
})

CFG.TEST = EasyDict({
    'INTERVAL': 1,
    'DATA': EasyDict({'BATCH_SIZE': 32}),
})
