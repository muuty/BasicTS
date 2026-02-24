"""
Run STAEformer baseline experiments with synthetic corruption injection.

Corruption types:
  A. channel_stuck  - speed channel fixed at 0 (mimics broken speed sensor)
  B. noisy          - Gaussian noise on physical channels
  C. intermittent_missing - random timesteps zeroed out (scattered)

For each type, runs corruption rates: 10%, 20%, 30%, 50%.
Same rate → same corrupted nodes (for fair cross-type comparison).

Usage:
    # Run all experiments (adds to ray queue)
    python experiments/run_corruption_baseline.py

    # Run specific type/rate
    python experiments/run_corruption_baseline.py --type channel_stuck --rate 0.1

    # Dry run (print configs without launching)
    python experiments/run_corruption_baseline.py --dry-run
"""

import os
import sys
import argparse
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def make_config(corruption_type: str, corruption_rate: float, gpu: str = '0'):
    """Generate a full training config dict for a corruption experiment."""
    from easydict import EasyDict
    from basicts.metrics import masked_mae, masked_mape, masked_rmse
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.runners import SimpleTimeSeriesForecastingRunner
    from basicts.scaler import ZScoreScaler
    from basicts.utils import get_regular_settings
    from baselines.STAEformer.arch import STAEformer

    DATA_NAME = 'SAN_BERNARDINO'
    regular_settings = get_regular_settings(DATA_NAME)
    INPUT_LEN = regular_settings['INPUT_LEN']
    OUTPUT_LEN = regular_settings['OUTPUT_LEN']
    TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
    NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
    RESCALE = regular_settings['RESCALE']
    NULL_VAL = regular_settings['NULL_VAL']
    NUM_NODES = 893

    # Corruption config
    corruption = {
        'type': corruption_type,
        'rate': corruption_rate,
        'seed': 42,
    }
    # Type-specific defaults
    if corruption_type == 'channel_stuck':
        corruption['stuck_channel'] = 2  # speed
    elif corruption_type == 'flow_stuck':
        corruption['type'] = 'channel_stuck'  # reuse channel_stuck logic
        corruption['stuck_channel'] = 0  # flow
    elif corruption_type == 'noisy':
        corruption['noise_ratio'] = 0.3
        corruption['channels'] = [0, 1, 2]
    elif corruption_type == 'intermittent_missing':
        corruption['missing_rate'] = 0.1
        corruption['channels'] = [0, 1, 2]

    rate_str = str(int(corruption_rate * 100))
    tag = f"corrupt_{corruption_type}_r{rate_str}"

    CFG = EasyDict()
    CFG.DESCRIPTION = f'STAEformer 5ch + {corruption_type} corruption rate={rate_str}%'
    CFG.GPU_NUM = 1
    CFG.RUNNER = SimpleTimeSeriesForecastingRunner

    CFG.DATASET = EasyDict()
    CFG.DATASET.NAME = DATA_NAME
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
        'data_range': (0, 26280),
        'corruption': corruption,
    })

    CFG.SCALER = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_ratio': TRAIN_VAL_TEST_RATIO[0],
        'norm_each_channel': NORM_EACH_CHANNEL,
        'rescale': RESCALE,
    })

    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = 'STAEformer'
    CFG.MODEL.ARCH = STAEformer
    CFG.MODEL.PARAM = {
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
    CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
    CFG.MODEL.TARGET_FEATURES = [0]

    CFG.METRICS = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({
        'MAE': masked_mae,
        'MAPE': masked_mape,
        'RMSE': masked_rmse,
    })
    CFG.METRICS.TARGET = 'MAE'
    CFG.METRICS.NULL_VAL = NULL_VAL

    NUM_EPOCHS = 30
    CFG.TRAIN = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints',
        f'STAEformer_5ch_{tag}',
        f'{DATA_NAME}_{NUM_EPOCHS}_{INPUT_LEN}_{OUTPUT_LEN}'
    )
    CFG.TRAIN.LOSS = masked_mae
    CFG.TRAIN.OPTIM = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = "Adam"
    CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0003}
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.1}
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

    return CFG


def _subdirs(path):
    """List immediate subdirectories."""
    if not os.path.exists(path):
        return []
    return [os.path.join(path, d) for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))]


CORRUPTION_TYPES = ['channel_stuck', 'flow_stuck', 'noisy', 'intermittent_missing']
CORRUPTION_RATES = [0.1, 0.2, 0.3, 0.5]


def main():
    parser = argparse.ArgumentParser(description='Run corruption baseline experiments')
    parser.add_argument('--type', type=str, choices=CORRUPTION_TYPES, default=None,
                        help='Corruption type (default: all)')
    parser.add_argument('--rate', type=float, choices=CORRUPTION_RATES, default=None,
                        help='Corruption rate (default: all)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--dry-run', action='store_true', help='Print configs without launching')
    args = parser.parse_args()

    types = [args.type] if args.type else CORRUPTION_TYPES
    rates = [args.rate] if args.rate else CORRUPTION_RATES

    experiments = [(t, r) for t in types for r in rates]

    print(f"=== Corruption Baseline Experiments ===")
    print(f"Types: {types}")
    print(f"Rates: {rates}")
    print(f"Total experiments: {len(experiments)}")
    print()

    if args.dry_run:
        for ctype, rate in experiments:
            cfg = make_config(ctype, rate, args.gpu)
            ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
            done = any(os.path.exists(os.path.join(d, 'test_metrics.json'))
                       for d in _subdirs(ckpt_dir))
            status = " [DONE]" if done else ""
            print(f"  [{ctype} r={int(rate*100)}%] -> {ckpt_dir}{status}")
        print("\nDry run complete. No experiments launched.")
        return

    from basicts import launch_training

    for i, (ctype, rate) in enumerate(experiments):
        cfg = make_config(ctype, rate, args.gpu)

        # Skip already-completed experiments
        ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
        if any(os.path.exists(os.path.join(d, 'test_metrics.json'))
               for d in _subdirs(ckpt_dir)):
            print(f"\n[{i+1}/{len(experiments)}] SKIP (already done): {ctype} rate={int(rate*100)}%")
            continue

        print(f"\n[{i+1}/{len(experiments)}] Launching: {ctype} rate={int(rate*100)}%")
        print(f"  Checkpoint: {ckpt_dir}")
        launch_training(cfg, gpus=args.gpu)
        print(f"  Done: {ctype} rate={int(rate*100)}%")

    print(f"\n=== All {len(experiments)} experiments complete ===")


if __name__ == '__main__':
    main()
