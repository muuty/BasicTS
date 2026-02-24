# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easytorch.config import init_cfg
import basicts


def get_new_ckpt_save_dir(cfg_path, exp_name):
    cfg = init_cfg(cfg_path)
    original_dir = cfg.TRAIN.CKPT_SAVE_DIR
    suffix = original_dir[len('checkpoints/'):]
    return f'checkpoints/{exp_name}/{suffix}'

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STID/PEMS04.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    parser.add_argument('-n', '--exp_name', default=None, help='name of the experiment')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = init_cfg(args.cfg)
    if args.exp_name:
        cfg.TRAIN.CKPT_SAVE_DIR = get_new_ckpt_save_dir(args.cfg, args.exp_name)
        print(f" new dir: {cfg.TRAIN.CKPT_SAVE_DIR}")
    basicts.launch_training(cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()
