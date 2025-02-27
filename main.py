from easytorch.config import init_cfg
from basicts.utils import get_regular_settings


import basicts
import os
if __name__ == '__main__':
    datasets = ('METR-LA', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMS-BAY')
    for dataset in datasets:
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for exp_num in range(5):
                dataset = 'PEMS04'
                regular_settings = get_regular_settings(dataset)
                INPUT_LEN = regular_settings['INPUT_LEN']
                OUTPUT_LEN = regular_settings['OUTPUT_LEN']

                CFG = init_cfg(f"baselines/STGCN/{dataset}.py", True)
                CFG.DEVICE_NUM = 0

                # SELECTION
                CFG.TRAIN.DATA.SELECTION_RATIO = 0.01
                CFG.TRAIN.DATA.SELECTION_STRATEGY = 'random'
                CFG.TRAIN.DATA.EMBEDDING_STRATEGY = None

                # # OUTLIER
                # CFG.TRAIN.DATA.OUTLIER_REMOVAL_EMBEDDING_STRATEGY = 'umap'
                # CFG.TRAIN.DATA.OUTLIER_REMOVAL_STRATEGY = 'hdbscan'
                # CFG.TRAIN.DATA.OUTLIER_REMOVAL_RATIO = ratio
                CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
                    'checkpoints',
                    CFG.MODEL.NAME,
                    '_'.join([dataset, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
                    f"{CFG.TRAIN.DATA.SELECTION_STRATEGY}",f"ratio-{CFG.TRAIN.DATA.SELECTION_RATIO}", str(exp_num)
                )
                basicts.launch_training(CFG, 'mps', node_rank=0)