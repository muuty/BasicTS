"""
Ablation B2: hidden_dim=32 with residual connection.
Same as v2 except hidden_dim=32 (same as v1 size).
Resolves v1/v2 confound: isolates residual contribution from capacity increase.
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

from ..arch.denoising_pretrain_model import DenoisingPretrainModel
from ..loss.denoising_loss import get_denoising_loss

DATA_NAME = 'SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = DenoisingPretrainModel
NUM_NODES = 893

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "input_dim": 5,
    "output_dim": 1,
    "d_model": 5,
    "hidden_dim": 32,          # <-- v1 size
    "temporal_layers": 4,
    "spatial_layers": 1,
    "k_neighbors": 10,
    "dropout": 0.1,
    "adj_path": "datasets/SAN_BERNARDINO/adj_mx.pkl",
    "noise_rate_range": (0.1, 0.7),
    "noise_severity_range": (0.1, 1.5),
    "physical_channels": [0, 1, 2],
    "residual_connection": True,  # <-- residual ON (unlike v1)
}
NUM_EPOCHS = 30

CFG = EasyDict()
CFG.DESCRIPTION = 'Ablation B2: hidden=32 + residual (v1 capacity, v2 design)'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    'data_range': (0, 26280),
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'AblationB2_Hidden32'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'MAPE': masked_mape, 'RMSE': masked_rmse})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join('checkpoints', CFG.MODEL.NAME, '_'.join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]))
CFG.TRAIN.LOSS = get_denoising_loss(passthrough_weight=1.0)
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

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
