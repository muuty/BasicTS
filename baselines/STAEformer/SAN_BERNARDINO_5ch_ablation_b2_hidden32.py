"""
Ablation B2 Downstream: hidden_dim=32 with residual + noise augmentation.
Resolves v1/v2 confound: same capacity as v1, but with residual connection.
"""
import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.runner.noisy_representation_learning_runner import NoisyRepresentationLearningRunner

DATA_NAME = 'SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = STAEformer
NUM_NODES = 893

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 288,
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
}
NUM_EPOCHS = 30

CFG = EasyDict()
CFG.DESCRIPTION = 'Ablation B2: hidden=32 + residual + noise augmentation'
CFG.GPU_NUM = 1
CFG.RUNNER = NoisyRepresentationLearningRunner

CFG.ENCODER = {
    'type': 'DenoisingEncoder',
    'source': 'pretrained',
    'freeze': True,
    'ckpt_path': 'checkpoints/AblationB2_Hidden32/SAN_BERNARDINO_30_12_12/b541eefca71c78c42561ef3e99a60594/AblationB2_Hidden32_best_val_MAE.pt',
    'input_dim': 5,
    'd_model': 5,
    'hidden_dim': 32,
    'temporal_layers': 4,
    'spatial_layers': 1,
    'k_neighbors': 10,
    'dropout': 0.1,
    'adj_path': 'datasets/SAN_BERNARDINO/adj_mx.pkl',
    'physical_channels': [0, 1, 2],
    'residual_connection': True,
    'include_tod_dow': False,
}

CFG.NOISE_AUGMENTATION = EasyDict({
    'prob': 0.5,
    'rate_range': [0.05, 0.3],
    'types': ['gaussian', 'bias', 'stuck', 'drift', 'dead'],
    'severity_range': [0.1, 0.5],
    'physical_channels': [0, 1, 2],
})

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
CFG.MODEL.NAME = 'STAEformer_ablation_b2_hidden32'
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
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0003}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.1}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
