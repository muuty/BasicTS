# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
"""
JY version: Supports BASICTS_OVERRIDE_JSON environment variable for runtime config overrides.
"""
import os
import sys
import json
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


def apply_overrides(cfg, overrides: dict):
    """
    Apply dotted-key overrides to the config object.
    e.g., {"TRAIN.CKPT_SAVE_DIR": "/new/path", "IL.NUM_CLIENTS": 10}
    """
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                # Create attribute if it doesn't exist
                print(f"[WARN] Config path '{key}' not found, skipping.")
                break
        else:
            final_key = parts[-1]
            if hasattr(obj, final_key):
                setattr(obj, final_key, value)
            elif isinstance(obj, dict):
                obj[final_key] = value
            else:
                print(f"[WARN] Cannot set '{key}' = {value}, target is not dict or object.")
    return cfg


def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STID/PEMS04.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    parser.add_argument('-n', '--exp_name', default=None, help='name of the experiment')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = init_cfg(args.cfg)

    # Apply exp_name override (legacy support)
    if args.exp_name:
        cfg.TRAIN.CKPT_SAVE_DIR = get_new_ckpt_save_dir(args.cfg, args.exp_name)
        print(f" new dir: {cfg.TRAIN.CKPT_SAVE_DIR}")

    # Apply BASICTS_OVERRIDE_JSON environment variable
    override_json = os.environ.get("BASICTS_OVERRIDE_JSON", None)
    if override_json:
        try:
            overrides = json.loads(override_json)
            print(f"[INFO] Applying {len(overrides)} overrides from BASICTS_OVERRIDE_JSON")
            cfg = apply_overrides(cfg, overrides)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse BASICTS_OVERRIDE_JSON: {e}")

    basicts.launch_training(cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()

