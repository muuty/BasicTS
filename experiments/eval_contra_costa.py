"""Quick noise eval for CONTRA_COSTA baseline and denoising models."""
import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj
from basicts.runners.noise_eval import inject_noise, select_corrupt_nodes, DEFAULT_NOISE_CONFIGS, SEED

DEVICE = 'cuda:1'
DATA_NAME = 'CONTRA_COSTA'
NUM_NODES = 773

# Load functional indices
dead = np.load(f'datasets/{DATA_NAME}/dead_indices.npy')
major = np.load(f'datasets/{DATA_NAME}/major_fail_indices.npy')
functional = np.setdiff1d(np.arange(NUM_NODES), np.union1d(dead, major))
print(f'Nodes: {NUM_NODES} total, {len(functional)} functional')

# Dataset & scaler
rs = get_regular_settings(DATA_NAME)
test_dataset = TimeSeriesForecastingDataset(dataset_name=DATA_NAME, train_val_test_ratio=rs['TRAIN_VAL_TEST_RATIO'],
                                            mode='test', input_len=12, output_len=12, data_range=(0, 26280))
loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

scaler = ZScoreScaler(dataset_name=DATA_NAME, train_ratio=rs['TRAIN_VAL_TEST_RATIO'][0],
                       norm_each_channel=rs['NORM_EACH_CHANNEL'], rescale=rs['RESCALE'])

# Pre-compute corrupt nodes for each rate
noise_configs = DEFAULT_NOISE_CONFIGS
rates = set(c[1] for c in noise_configs)
corrupt_sets = {}
for rate in rates:
    corrupt, healthy = select_corrupt_nodes(NUM_NODES, rate, functional, SEED)
    corrupt_sets[rate] = (corrupt, healthy)

physical_channels = [0, 1, 2]
forward_features = [0, 1, 2, 3, 4]


def load_staeformer(ckpt_path):
    from baselines.STAEformer.arch import STAEformer
    model = STAEformer(num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
                       input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
                       dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=24,
                       feed_forward_dim=256, num_heads=4, num_layers=1, dropout=0.1, use_mixed_proj=True)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_encoder(downstream_ckpt):
    """Load encoder from downstream checkpoint (has encoder_state_dict + encoder_config)."""
    from baselines.ContextContrastive.arch import build_encoder
    state = torch.load(downstream_ckpt, map_location=DEVICE)
    encoder_cfg = state['encoder_config']
    encoder_cfg['adj_path'] = f'datasets/{DATA_NAME}/adj_mx.pkl'
    encoder = build_encoder(encoder_cfg)
    encoder.load_state_dict(state['encoder_state_dict'])
    return encoder.to(DEVICE).eval()


def evaluate(model, encoder=None):
    results = {}

    # Clean eval
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            # Raw inputs (unnormalized)
            raw_inputs = batch['inputs'][:, :, :, forward_features].to(DEVICE)
            raw_target = batch['target'][:, :, :, :1].to(DEVICE)

            # Normalize for model
            inputs_norm = scaler.transform(raw_inputs)

            if encoder is not None:
                model_input = encoder.encode(inputs_norm)
            else:
                model_input = inputs_norm

            pred = model(history_data=model_input, future_data=None, batch_seen=0, epoch=0, train=False)
            if isinstance(pred, dict):
                pred = pred['prediction']

            # Inverse transform predictions only (target is already raw)
            pred_rs = scaler.inverse_transform(pred)
            all_preds.append(pred_rs.cpu().numpy())
            all_targets.append(raw_target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_targets, axis=0)

    # Per-node MAE on functional (observed entries only)
    mask = (tgts != 0).astype(float)
    err = np.abs(preds - tgts) * mask
    node_mae = err[:, :, :, 0].sum(axis=(0, 1)) / (mask[:, :, :, 0].sum(axis=(0, 1)) + 1e-8)
    clean_func_mae = node_mae[functional].mean()
    print(f'  Clean MAE (functional): {clean_func_mae:.4f}')
    results['clean'] = float(clean_func_mae)

    # Noise eval
    for noise_type, rate, severity, label in noise_configs:
        corrupt, healthy = corrupt_sets[rate]
        all_preds_n = []
        rng_local = np.random.RandomState(SEED)

        with torch.no_grad():
            for batch in loader:
                raw_inputs = batch['inputs'][:, :, :, forward_features].to(DEVICE)

                # Inject noise on RAW data (before normalization)
                noisy_raw = inject_noise(raw_inputs, noise_type, corrupt, severity, physical_channels, rng_local)

                # Normalize noisy data
                noisy_norm = scaler.transform(noisy_raw)

                if encoder is not None:
                    model_input = encoder.encode(noisy_norm)
                else:
                    model_input = noisy_norm

                pred = model(history_data=model_input, future_data=None, batch_seen=0, epoch=0, train=False)
                if isinstance(pred, dict):
                    pred = pred['prediction']

                pred_rs = scaler.inverse_transform(pred)
                all_preds_n.append(pred_rs.cpu().numpy())

        preds_n = np.concatenate(all_preds_n, axis=0)
        err_n = np.abs(preds_n - tgts) * mask
        node_mae_n = err_n[:, :, :, 0].sum(axis=(0, 1)) / (mask[:, :, :, 0].sum(axis=(0, 1)) + 1e-8)

        func_mae_n = float(node_mae_n[functional].mean())
        healthy_mae_n = float(node_mae_n[healthy].mean())
        healthy_mae_c = float(node_mae[healthy].mean())

        func_deg = (func_mae_n - clean_func_mae) / clean_func_mae * 100
        healthy_deg = (healthy_mae_n - healthy_mae_c) / healthy_mae_c * 100

        print(f'  [{label}] func: {clean_func_mae:.2f}->{func_mae_n:.2f} ({func_deg:+.1f}%) | healthy spill: {healthy_deg:+.1f}%')
        results[label] = {'func_deg': float(func_deg), 'healthy_deg': float(healthy_deg), 'func_mae': func_mae_n}

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['baseline', 'denoising', 'noisy', 'denoising_noisy', 'denoising_v2', 'denoising_v2_noisy'],
                        help='Models to evaluate')
    args = parser.parse_args()

    results_path = 'experiments/noise_vulnerability_results/contra_costa_results.json'
    os.makedirs('experiments/noise_vulnerability_results', exist_ok=True)

    # Load existing results
    if os.path.exists(results_path):
        with open(results_path) as f:
            output = json.load(f)
    else:
        output = {}

    if 'baseline' in args.models:
        print('\n' + '=' * 60)
        print('CONTRA_COSTA: STAEformer 5ch Baseline')
        print('=' * 60)
        ckpt = glob.glob('checkpoints/STAEformer_5ch/CONTRA_COSTA_30_12_12/*/STAEformer_5ch_best_val_MAE.pt')[0]
        model = load_staeformer(ckpt)
        output['baseline'] = evaluate(model)
        del model; torch.cuda.empty_cache()

    if 'denoising' in args.models:
        print('\n' + '=' * 60)
        print('CONTRA_COSTA: STAEformer 5ch + Denoising')
        print('=' * 60)
        ckpt = glob.glob('checkpoints/STAEformer_5ch_denoising/CONTRA_COSTA_30_12_12/*/STAEformer_5ch_denoising_best_val_MAE.pt')[0]
        model = load_staeformer(ckpt)
        encoder = load_encoder(ckpt)
        output['denoising'] = evaluate(model, encoder)
        del model, encoder; torch.cuda.empty_cache()

    if 'noisy' in args.models:
        print('\n' + '=' * 60)
        print('CONTRA_COSTA: STAEformer 5ch + Noisy Training')
        print('=' * 60)
        ckpt = glob.glob('checkpoints/STAEformer_5ch_noisy/CONTRA_COSTA_30_12_12/*/STAEformer_5ch_noisy_best_val_MAE.pt')[0]
        model = load_staeformer(ckpt)
        output['noisy'] = evaluate(model)
        del model; torch.cuda.empty_cache()

    if 'denoising_noisy' in args.models:
        print('\n' + '=' * 60)
        print('CONTRA_COSTA: STAEformer 5ch + Denoising + Noisy Training')
        print('=' * 60)
        ckpt = glob.glob('checkpoints/STAEformer_5ch_denoising_noisy/CONTRA_COSTA_30_12_12/*/STAEformer_5ch_denoising_noisy_best_val_MAE.pt')[0]
        model = load_staeformer(ckpt)
        encoder = load_encoder(ckpt)
        output['denoising_noisy'] = evaluate(model, encoder)
        del model, encoder; torch.cuda.empty_cache()

    if 'denoising_v2' in args.models:
        print('\n' + '=' * 60)
        print('CONTRA_COSTA: STAEformer 5ch + Denoising v2 (residual+h64)')
        print('=' * 60)
        ckpt = glob.glob('checkpoints/STAEformer_5ch_denoising_v2/CONTRA_COSTA_30_12_12/*/STAEformer_5ch_denoising_v2_best_val_MAE.pt')[0]
        model = load_staeformer(ckpt)
        encoder = load_encoder(ckpt)
        output['denoising_v2'] = evaluate(model, encoder)
        del model, encoder; torch.cuda.empty_cache()

    if 'denoising_v2_noisy' in args.models:
        print('\n' + '=' * 60)
        print('CONTRA_COSTA: STAEformer 5ch + Denoising v2 + Noisy Training')
        print('=' * 60)
        ckpt = glob.glob('checkpoints/STAEformer_5ch_denoising_v2_noisy/CONTRA_COSTA_30_12_12/*/STAEformer_5ch_denoising_v2_noisy_best_val_MAE.pt')[0]
        model = load_staeformer(ckpt)
        encoder = load_encoder(ckpt)
        output['denoising_v2_noisy'] = evaluate(model, encoder)
        del model, encoder; torch.cuda.empty_cache()

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {results_path}')
