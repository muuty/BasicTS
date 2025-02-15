from baselines.STGCN.PEMS03 import *
from datetime import datetime
print(CFG)

import basicts
import os
if __name__ == '__main__':
    for ratio in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for exp_num in range(0,5):
            CFG.DEVICE_NUM = 0
            CFG.TRAIN.DATA.SELECTION_RATIO = ratio
            CFG.TRAIN.DATA.SELECTION_STRATEGY = 'k_center_greedy'
            CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
                'checkpoints',
                MODEL_ARCH.__name__,
                '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
                CFG.TRAIN.DATA.SELECTION_STRATEGY, f"ration-{ratio}", str(exp_num)
            )
            basicts.launch_training(CFG, 'mps', node_rank=0)