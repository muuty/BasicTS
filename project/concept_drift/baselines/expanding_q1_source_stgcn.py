"""STGCN source model: 558 nodes on 2023 Q1 for expanding sensor experiment."""
import os
import sys
import numpy as np
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import masked_mape, unmasked_mae, unmasked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.STGCN.arch import STGCNNodeEmb

DATA_NAME = "expanding_experiment/SAN_BERNARDINO_2023_Q1"
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NODE_INDICES = np.load("datasets/expanding_experiment/nodes_q1.npy")
NUM_NODES = len(NODE_INDICES)  # 558

# Load and filter adjacency to Q1 nodes
adj_mx_raw, _ = load_adj("datasets/SAN_BERNARDINO/adj_mx.pkl", "normlap")
# adj_mx_raw[0] is the normalized laplacian (893, 893)
# Filter to Q1 node subset
adj_full = adj_mx_raw[0]
adj_filtered = adj_full[np.ix_(NODE_INDICES, NODE_INDICES)]
adj_mx = torch.Tensor(adj_filtered)

NODE_INDICES = NODE_INDICES.tolist()

MODEL_PARAM = {
    "Ks": 3,
    "Kt": 3,
    "blocks": [[5], [64, 16, 64], [64, 16, 64], [128, 128], [OUTPUT_LEN]],
    "T": INPUT_LEN,
    "num_nodes": NUM_NODES,
    "act_func": "glu",
    "graph_conv_type": "cheb_graph_conv",
    "adj_matrix": adj_mx,
    "bias": True,
    "droprate": 0.5,
    "node_emb_dim": 24,
}

CFG = EasyDict()
CFG.DESCRIPTION = "STGCNNodeEmb Source 558 nodes 5ch - 2023 Q1 (expanding experiment)"
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
CFG.MODEL.NAME = STGCNNodeEmb.__name__
CFG.MODEL.ARCH = STGCNNodeEmb
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
    "checkpoints", "Expanding_Source_STGCN",
    "_".join(["SAN_BERNARDINO_2023_Q1", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = unmasked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.0004, "weight_decay": 0.0003}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.5}
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
