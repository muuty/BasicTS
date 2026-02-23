# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import basicts
from easytorch.config import import_config, init_cfg

def get_new_ckpt_save_dir(cfg_path: str, exp_name: str) -> str:
    cfg = init_cfg(cfg_path)
    original_dir = cfg.TRAIN.CKPT_SAVE_DIR
    suffix = original_dir[len('checkpoints/'):]
    return f'checkpoints/{exp_name}/{suffix}'

def prepare_and_launch(cfg_path: str, gpus: str = "0", run: int = None, exp: str = None) -> None:
    """
    Config를 로드하고, run/exp에 따라 CKPT_SAVE_DIR을 수정한 후 training을 실행합니다.
    
    Args:
        cfg_path: Config 파일 경로
        gpus: 사용할 GPU ID (기본값: "0")
        run: Run 인덱스 (None이면 무시)
        exp: 실험 이름 (None이면 무시)
    """
    cfg = import_config(cfg_path, verbose=False)

    if exp is not None:
        cfg['TRAIN']['CKPT_SAVE_DIR'] = get_new_ckpt_save_dir(cfg_path, exp)
        print(f" new dir: {cfg['TRAIN']['CKPT_SAVE_DIR']}")

    if run is not None:
        if 'TRAIN' in cfg and 'CKPT_SAVE_DIR' in cfg['TRAIN']:
            cfg['TRAIN']['CKPT_SAVE_DIR'] = os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], str(run))

    
    basicts.launch_training(cfg, gpus, node_rank=0,)

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STGCN/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    parser.add_argument('-r', '--run', type=int, default=None, help='run index for multiple experiments')
    parser.add_argument('-e', '--exp', type=str, default=None, help='exp name')
    return parser.parse_args()

def main():
    args = parse_args()
    prepare_and_launch(args.cfg, args.gpus, args.run, args.exp)


if __name__ == '__main__':
    main()
