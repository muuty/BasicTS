"""AGCRN source model: 558 nodes on 2023 Q1 for expanding sensor experiment."""
import os
import sys
import numpy as np
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import masked_mape, unmasked_mae, unmasked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler

from baselines.AGCRN.arch import AGCRN

DATA_NAME = "expanding_experiment/SAN_BERNARDINO_2023_Q1"
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NODE_INDICES = np.load("datasets/expanding_experiment/nodes_q1.npy").tolist()
NUM_NODES = len(NODE_INDICES)  # 558

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_dim": 1,
    "rnn_units": 64,
    "output_dim": 1,
    "horizon": OUTPUT_LEN,
    "num_layers": 2,
    "default_graph": True,
    "embed_dim": 10,
    "cheb_k": 2,
}

CFG = EasyDict()
CFG.DESCRIPTION = "AGCRN Source 558 nodes - 2023 Q1 (expanding experiment)"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "node_indices": NODE_INDICES,
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
CFG.MODEL.NAME = AGCRN.__name__
CFG.MODEL.ARCH = AGCRN
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
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
    "checkpoints", "Expanding_Source_AGCRN",
    "_".join(["SAN_BERNARDINO_2023_Q1", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = unmasked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.003}
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
