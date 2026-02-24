"""MultiTask STAEformer: Forecasting + Cross-Variable Reconstruction (lambda=1.0)

No encoder bottleneck. STAEformer directly processes 5 raw features.
Auxiliary reconstruction head forces cross-variable understanding.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/multitask_crossvar_lambda10.py', gpus='1')"
"""
import os, sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.ContextContrastive.arch import MultiTaskSTAEformer
from baselines.ContextContrastive.loss import get_multitask_crossvar_loss

DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN, OUTPUT_LEN = regular_settings['INPUT_LEN'], regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL, RESCALE, NULL_VAL = regular_settings['NORM_EACH_CHANNEL'], regular_settings['RESCALE'], regular_settings['NULL_VAL']

NUM_NODES = 893
LAMBDA_RECON = 1.0

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "input_dim": 5,          # flow, occ, speed, tod, dow
    "output_dim": 1,
    "steps_per_day": 288,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "use_mixed_proj": True,
    # Multi-task specific
    "tod_feat_idx": 3,
    "dow_feat_idx": 4,
    "physical_feat_indices": [0, 1, 2],
}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = f'MultiTask STAEformer: forecast + cross-var recon (lambda={LAMBDA_RECON})'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

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
    'NAME': f'MultiTask_crossvar_lambda{LAMBDA_RECON}_3mo',
    'ARCH': MultiTaskSTAEformer,
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
    'LOSS': get_multitask_crossvar_loss(lambda_recon=LAMBDA_RECON),
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
    'DATA': EasyDict({'BATCH_SIZE': 64}),
})
