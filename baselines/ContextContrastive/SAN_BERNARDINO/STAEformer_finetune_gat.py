"""
Fine-tuning with Top-K Sparse GAT Pre-trained Encoder

Uses the pre-trained SpatioTemporalEncoder (GAT-based) for downstream traffic prediction.

Prerequisites:
    Run pre-training first: pretrain_gat.py

Usage:
    python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/STAEformer_finetune_gat.py', gpus='0')"
"""
import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.runner import RepresentationLearningRunner

############################## Hot Parameters ##############################
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Pre-trained encoder config (must match pre-training)
D_MODEL = 64
TEMPORAL_LAYERS = 2
TEMPORAL_HEADS = 4
SPATIAL_LAYERS = 1
SPATIAL_HEADS = 4
K_NEIGHBORS = 10

# Fine-tuning parameters
ENCODER_LR = 1e-5  # Lower LR for encoder
DOWNSTREAM_LR = 1e-3  # Standard LR for downstream model

# Load adjacency matrix for GAT encoder
adj_mx, _ = load_adj("datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl", "doubletransition")
adj_mx = torch.Tensor(adj_mx[0])

# STAEformer model
MODEL_ARCH = STAEformer
NUM_NODES = 893

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 288,
    "input_dim": D_MODEL + 2,  # encoded (D_MODEL) + tod + dow
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
NUM_EPOCHS = 100  # Max epochs (early stopping will handle actual stopping)

############################## Encoder Configuration (GAT) ##############################
ENCODER = {
    'type': 'TransformerEncoder',
    'source': 'pretrained',
    'encoder_type': 'SpatioTemporalEncoder',  # Use GAT-based encoder
    'ckpt_path': f'checkpoints/ContextContrastive_GAT_pretrain_3mo/xtraffic_SAN_BERNARDINO_30_{INPUT_LEN}_{OUTPUT_LEN}/*/GAT_Pretrain_best_val_MAE.pt',
    'input_dim': 3,
    'd_model': D_MODEL,
    'temporal_layers': TEMPORAL_LAYERS,
    'temporal_heads': TEMPORAL_HEADS,
    'spatial_layers': SPATIAL_LAYERS,
    'spatial_heads': SPATIAL_HEADS,
    'k_neighbors': K_NEIGHBORS,
    'dropout': 0.1,
    'adj_matrix': adj_mx,  # Pass adjacency matrix
    'include_tod_dow': True,
    # Fine-tuning settings
    'freeze': False,  # Fine-tune the encoder
    'lr': ENCODER_LR,
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Fine-tune with Top-K Sparse GAT Pre-trained Encoder'
CFG.GPU_NUM = 1
CFG.RUNNER = RepresentationLearningRunner
CFG.ENCODER = ENCODER

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
CFG.MODEL.NAME = 'STAEformer_GAT'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 3, 4]
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
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'ContextContrastive_finetune_GAT_3mo',
    '_'.join([DATA_NAME.replace('/', '_'), str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": DOWNSTREAM_LR, "weight_decay": 0.0003}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.1}

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
CFG.TEST.INCIDENT_METADATA_PATH = f"datasets/{DATA_NAME}/incident_metadata_2023.csv"

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
