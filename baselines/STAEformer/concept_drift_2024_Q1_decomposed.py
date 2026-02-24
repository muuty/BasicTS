import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import masked_mape, unmasked_mae, unmasked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import DecomposedRunner
from basicts.scaler import ZScoreScaler

from baselines.STAEformer.arch.decomposed_staeformer import DecomposedSTAEformer
from baselines.STAEformer.loss import decomposed_mae

DATA_NAME = "SAN_BERNARDINO_2024_Q1"
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NUM_NODES = 893

BACKBONE_PARAMS = {
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

MODEL_PARAM = {
    "backbone_params": BACKBONE_PARAMS,
    "scale_head_hidden": 32,
}

CFG = EasyDict()
CFG.DESCRIPTION = "DecomposedSTAEformer (scale head + residual backbone) - 2024 Q1"
CFG.GPU_NUM = 1
CFG.RUNNER = DecomposedRunner

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_ratio": TRAIN_VAL_TEST_RATIO[0],
    "norm_each_channel": False,
    "rescale": True,
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "DecomposedSTAEformer"
CFG.MODEL.ARCH = DecomposedSTAEformer
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    "MAE": unmasked_mae,
    "RMSE": unmasked_rmse,
    "MAPE": masked_mape,
})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = 0.0

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = 30
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "ConceptDrift_Decomposed",
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = decomposed_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0003,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20, 25],
    "gamma": 0.1
}
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
