"""Control: Scratch Encoder (random init) -> STAEformer

Same TransformerEncoder(d_model=32) architecture as cross-variable pretrained,
but with random initialization. This isolates the effect of pre-training.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/STAEformer_scratch_5feat_cross_variable_downstream.py', gpus='0')"
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

NUM_NODES, D_MODEL = 893, 32

CFG_ENCODER = {
    'type': 'TransformerEncoder',
    'source': 'scratch',
    'freeze': False,
    'input_dim': 5,
    'd_model': D_MODEL,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,
    'dim_feedforward': D_MODEL * 4,
    'include_tod_dow': True,
    'tod_idx': 3,
    'dow_idx': 4,
}

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "input_dim": D_MODEL + 2,  # encoder(32) + tod + dow
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
CFG.DESCRIPTION = 'Control: Scratch Encoder (random init) -> Downstream (5feat, d=32)'
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
    'NAME': 'STAEformer_5feat_scratch_encoder_downstream_3mo',
    'ARCH': STAEformer,
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
