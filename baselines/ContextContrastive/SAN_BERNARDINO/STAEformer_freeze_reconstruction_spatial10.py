"""STAEformer with FROZEN Reconstruction-Pretrained Encoder (Spatial 10%)

Uses encoder pretrained with MAE-style reconstruction loss (learnable mask token).
No false negative problem - direct supervision signal.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/STAEformer_freeze_reconstruction_spatial10.py', gpus='0')"
"""
import os, sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.runner import RepresentationLearningRunner

DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN, OUTPUT_LEN = regular_settings['INPUT_LEN'], regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL, RESCALE, NULL_VAL = regular_settings['NORM_EACH_CHANNEL'], regular_settings['RESCALE'], regular_settings['NULL_VAL']

NUM_NODES, D_MODEL = 893, 64

CFG_ENCODER = {
    'type': 'TransformerEncoder',
    'source': 'pretrained',
    'freeze': True,
    'ckpt_path': 'checkpoints/ContextContrastive_reconstruction_spatial10_3mo/xtraffic_SAN_BERNARDINO_12_12/*/ContextContrastive_reconstruction_spatial10_3mo_best_val*MAE.pt',
    'input_dim': 3,
    'd_model': D_MODEL,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,
    'include_tod_dow': True,
}

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "input_dim": D_MODEL + 2,
    "output_dim": 1,
    "steps_per_day": 288,
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
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = 'STAEformer + FROZEN Reconstruction Encoder (Spatial 10%, MAE-style)'
CFG.GPU_NUM = 1
CFG.RUNNER = RepresentationLearningRunner
CFG.ENCODER = CFG_ENCODER

CFG.DATASET = EasyDict({
    'NAME': DATA_NAME,
    'TYPE': TimeSeriesForecastingDataset,
    'PARAM': EasyDict({
        'dataset_name': DATA_NAME,
        'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
    })
})

CFG.SCALER = EasyDict({
    'TYPE': ZScoreScaler,
    'PARAM': EasyDict({
        'dataset_name': DATA_NAME,
        'train_ratio': TRAIN_VAL_TEST_RATIO[0],
        'norm_each_channel': NORM_EACH_CHANNEL,
        'rescale': RESCALE,
    })
})

CFG.MODEL = EasyDict({
    'NAME': 'ContextContrastive_freeze_reconstruction_spatial10_3mo',
    'ARCH': STAEformer,
    'PARAM': MODEL_PARAM,
    'FORWARD_FEATURES': [0, 3, 4],
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
    'LOSS': masked_mae,
    'OPTIM': EasyDict({
        'TYPE': 'Adam',
        'PARAM': {'lr': 0.002, 'weight_decay': 1e-5},
    }),
    'LR_SCHEDULER': EasyDict({
        'TYPE': 'CosineAnnealingLR',
        'PARAM': {'T_max': NUM_EPOCHS, 'eta_min': 1e-6},
    }),
    'CLIP_GRAD_PARAM': {'max_norm': 5.0},
    'DATA': EasyDict({'BATCH_SIZE': 32, 'SHUFFLE': True}),
    'EARLY_STOPPING_PATIENCE': 20,
})

CFG.VAL = EasyDict({
    'INTERVAL': 1,
    'DATA': EasyDict({'BATCH_SIZE': 64}),
})

CFG.TEST = EasyDict({
    'INTERVAL': NUM_EPOCHS,
    'DATA': EasyDict({'BATCH_SIZE': 64}),
})
