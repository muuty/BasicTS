"""Spatial (Node) Masking 90% Pre-training"""
import os, sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from ..arch import ConfigurablePretrainModel
from ..loss import get_contrastive_loss

DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN, OUTPUT_LEN = regular_settings['INPUT_LEN'], regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL, RESCALE, NULL_VAL = regular_settings['NORM_EACH_CHANNEL'], regular_settings['RESCALE'], regular_settings['NULL_VAL']

NUM_NODES, D_MODEL, NUM_LAYERS, NHEAD, TEMPERATURE = 893, 64, 2, 4, 0.1
MASK_RATIO = 0.90

AUGMENTATION_CONFIG = {'node_masking': {'mask_ratio': MASK_RATIO}}
MODEL_PARAM = {"num_nodes": NUM_NODES, "input_len": INPUT_LEN, "output_len": OUTPUT_LEN, "input_dim": 3, "output_dim": 1,
               "d_model": D_MODEL, "num_layers": NUM_LAYERS, "nhead": NHEAD, "dropout": 0.1, "augmentation_config": AUGMENTATION_CONFIG}
NUM_EPOCHS = 30

CFG = EasyDict()
CFG.DESCRIPTION = f'Contrastive Pre-training: Spatial Masking {int(MASK_RATIO*100)}%'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET = EasyDict({'NAME': DATA_NAME, 'TYPE': TimeSeriesForecastingDataset,
    'PARAM': EasyDict({'dataset_name': DATA_NAME, 'train_val_test_ratio': TRAIN_VAL_TEST_RATIO, 'input_len': INPUT_LEN, 'output_len': OUTPUT_LEN, 'data_range': (0, 26280)})})
CFG.SCALER = EasyDict({'TYPE': ZScoreScaler, 'PARAM': EasyDict({'dataset_name': DATA_NAME, 'train_ratio': TRAIN_VAL_TEST_RATIO[0], 'norm_each_channel': NORM_EACH_CHANNEL, 'rescale': RESCALE})})
CFG.MODEL = EasyDict({'NAME': f'ContextContrastive_spatial_mask{int(MASK_RATIO*100)}pct_3mo', 'ARCH': ConfigurablePretrainModel, 'PARAM': MODEL_PARAM, 'FORWARD_FEATURES': [0, 3, 4], 'TARGET_FEATURES': [0]})
CFG.METRICS = EasyDict({'FUNCS': EasyDict({'MAE': masked_mae, 'MAPE': masked_mape, 'RMSE': masked_rmse}), 'TARGET': 'MAE', 'NULL_VAL': NULL_VAL})
CFG.TRAIN = EasyDict({'NUM_EPOCHS': NUM_EPOCHS, 'CKPT_SAVE_DIR': os.path.join('checkpoints', CFG.MODEL.NAME, '_'.join([DATA_NAME.replace('/', '_'), str(INPUT_LEN), str(OUTPUT_LEN)])),
    'LOSS': get_contrastive_loss(temperature=TEMPERATURE), 'OPTIM': EasyDict({'TYPE': 'Adam', 'PARAM': {'lr': 0.001, 'weight_decay': 1e-5}}),
    'LR_SCHEDULER': EasyDict({'TYPE': 'CosineAnnealingLR', 'PARAM': {'T_max': NUM_EPOCHS, 'eta_min': 1e-6}}), 'CLIP_GRAD_PARAM': {'max_norm': 5.0},
    'DATA': EasyDict({'BATCH_SIZE': 16, 'SHUFFLE': True})})
CFG.VAL = EasyDict({'INTERVAL': 1, 'DATA': EasyDict({'BATCH_SIZE': 16})})
CFG.TEST = EasyDict({'INTERVAL': NUM_EPOCHS, 'DATA': EasyDict({'BATCH_SIZE': 16})})
