from baselines.STGCN.PEMS03 import *
from datetime import datetime
print(CFG)
from selection.hdbscan_selection import HDBSCANSelection
from selection.random_selection import RandomSelection
from selection.k_random_greedy import KCenterGreedySelection
from selection.embedding.factory import get_embedding


import basicts
import os
if __name__ == '__main__':
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for exp_num in range(0,5):
            if exp_num==0 and ratio < 0.4:
                continue
            CFG.DEVICE_NUM = 0

            # SELECTION
            CFG.TRAIN.DATA.SELECTION_RATIO = ratio
            CFG.TRAIN.DATA.SELECTION_STRATEGY = 'random'
            CFG.TRAIN.DATA.EMBEDDING_STRATEGY = None

            # OUTLIER
            CFG.TRAIN.DATA.OUTLIER_REMOVAL_RUNNER = CFG.RUNNER
            CFG.TRAIN.DATA.OUTLIER_REMOVAL_STRATEGY = 'model_based'
            CFG.TRAIN.DATA.OUTLIER_REMOVAL_MODEL_PATH = 'checkpoints/STGCNChebGraphConv/PEMS03_100_12_12/k_center_greedy-raw/ration-0.2/0/52c71c8e73ad9255cdcf12d8dafde844/STGCNChebGraphConv_067.pt'
            CFG.TRAIN.DATA.OUTLIER_REMOVAL_RATIO = 0.1
            CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
                'checkpoints',
                MODEL_ARCH.__name__,
                '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
                f"{CFG.TRAIN.DATA.SELECTION_STRATEGY}-{CFG.TRAIN.DATA.EMBEDDING_STRATEGY}", f"ration-{ratio}", str(exp_num)
            )
            basicts.launch_training(CFG, 'mps', node_rank=0)
    # from baselines.STGCN.PEMS03 import *
    #
    # from easytorch.device import set_device_type
    # from easytorch.config import init_cfg
    #
    # CFG = init_cfg(CFG)
    # set_device_type("mps")
    # dataset = SimpleTimeSeriesForecastingRunner(CFG).build_train_dataset(CFG)
    # selection = HDBSCANSelection(dataset, ratio=0.1)
    # k_greedy_indices = selection.select_indices()