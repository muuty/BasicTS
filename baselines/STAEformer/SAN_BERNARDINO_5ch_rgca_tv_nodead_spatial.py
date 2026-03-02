"""
STAEformer 5ch + InputSpilloverCorrector V3 (Spatial-Temporal Reliability).
Dataset: SAN_BERNARDINO_NODEAD (773 nodes).

V3: temporal LinearAttention + spatial LinearAttention. No mean pool, no cross-attn.
Correction: delta × (1-r) — clean nodes get zero correction, noisy get full.

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_rgca_tv_nodead_spatial.py', gpus='1')"
"""
import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import unmasked_mae, unmasked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.runner.noisy_representation_learning_runner import NoisyRepresentationLearningRunner


def unmasked_identity_regularized_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    passthrough_loss: torch.Tensor = None,
) -> torch.Tensor:
    """Identity-regularized unmasked MAE. All targets contribute."""
    L_pred = torch.mean(torch.abs(prediction - target))
    if passthrough_loss is None:
        return L_pred
    return L_pred + passthrough_loss


############################## Hot Parameters ##############################
DATA_NAME = 'SAN_BERNARDINO_NODEAD'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']

MODEL_ARCH = STAEformer
NUM_NODES = 773

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
NUM_EPOCHS = 50

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STAEformer 5ch + V3 spatial-temporal reliability, NODEAD 773 nodes'
CFG.GPU_NUM = 1
CFG.RUNNER = NoisyRepresentationLearningRunner

############################## Encoder Configuration ##############################
CFG.ENCODER = {
    'type': 'InputSpilloverCorrector',
    'source': 'pretrained',
    'freeze': False,
    'lr': 1e-4,
    'ckpt_path': 'checkpoints/InputCorrectorPretrain_NODEAD_Spatial/SAN_BERNARDINO_NODEAD_30_12_12/*/InputCorrectorPretrain_NODEAD_Spatial_best_val_MAE.pt',
    'input_dim': 5,
    'd_model': 5,
    'hidden_dim': 64,
    'n_heads': 4,
    'temporal_layers': 2,
    'spatial_layers': 1,
    'dropout': 0.1,
    'physical_channels': [0, 1, 2],
    'residual_connection': True,
    'time_varying_correction': True,
}

############################## Noise Augmentation + Identity Constraint ##############################
CFG.NOISE_AUGMENTATION = EasyDict({
    'prob': 0.5,
    'rate_range': [0.05, 0.3],
    'types': ['gaussian', 'drift', 'dead', 'spike'],
    'severity_range': [0.1, 0.5],
    'physical_channels': [0, 1, 2],
    'identity_lambda': 0.1,
})

############################## Dataset Configuration ##############################
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
CFG.MODEL.NAME = 'STAEformer_5ch_rgca_tv_nodead_spatial'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': unmasked_mae,
    'RMSE': unmasked_rmse,
})
CFG.METRICS.TARGET = 'MAE'

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    CFG.MODEL.NAME,
    '_'.join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = unmasked_identity_regularized_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0003}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
CFG.EVAL.NOISE_ROBUSTNESS = True
