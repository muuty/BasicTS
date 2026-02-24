"""
Artificial Missing Pattern Shift Evaluation

Research question: How robust are different models to changes in missing data
patterns at test time? When the test-time missing pattern differs from training,
how does prediction quality degrade?

Models tested:
  1. STAEformer 5ch (flow+occ+speed+tod+dow) - can't distinguish missing from zero
  2. STAEformer 8ch mask (+ mask channels, masked_mae loss) - has mask but standard loss
  3. STAEformer 8ch mask-aware (+ mask channels + mask-aware loss)
  4. STGCN 1ch (flow only) - can't distinguish, GCN propagation
  5. STGCN 8ch mask-aware (8ch + mask-aware loss)

Shift types:
  1. Node death: r% of functional nodes -> all values zero at ALL timesteps
     (simulates permanent sensor failure at test time)
  2. Intermittent: p% of (timestep, functional_node) entries -> zero
     (simulates random data loss increase at test time)

For 5ch models: corruption = zero out physical channels (flow/occ/speed)
  -> model can't distinguish from real zero traffic
For 8ch models: corruption = zero physical + set mask channels to 0
  -> model knows which values are missing

Metric: healthy functional node MAE
  - Primary: target_mask==1 (observed entries only, from 8ch mask dataset)
  - Legacy: target>0 (for comparison, conflates missing with real zero)

Usage:
  python experiments/eval_missing_pattern_shift.py --gpu 1
  python experiments/eval_missing_pattern_shift.py --gpu 1 --models staeformer_5ch staeformer_8ch_mask_aware
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'missing_pattern_shift_results')
NUM_NODES = 893
SEED = 42

# ─── Shift Configurations ────────────────────────────────────────────
SHIFT_CONFIGS = [
    # (type, rate, label)
    ('node_death', 0.10, 'death_r10'),
    ('node_death', 0.20, 'death_r20'),
    ('node_death', 0.30, 'death_r30'),
    ('node_death', 0.50, 'death_r50'),
    ('intermittent', 0.10, 'intermit_p10'),
    ('intermittent', 0.20, 'intermit_p20'),
    ('intermittent', 0.30, 'intermit_p30'),
]


# ─── Model Loaders ───────────────────────────────────────────────────
def load_staeformer_5ch(device):
    from baselines.STAEformer.arch import STAEformer
    model = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt = 'checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/50637b58eb0e35770d311d9f7bdaa214/STAEformer_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_staeformer_5ch_maskloss(device):
    """5ch input + mask-aware loss (unified baseline).
    Same architecture as 5ch but trained with mask-aware loss on MASK dataset."""
    import glob
    from baselines.STAEformer.arch import STAEformer
    model = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt_pattern = 'checkpoints/STAEformer_5ch_maskloss/SAN_BERNARDINO_MASK_30_12_12/*/STAEformer_best_val_MAE.pt'
    ckpt = sorted(glob.glob(ckpt_pattern))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_staeformer_8ch_mask(device):
    from baselines.STAEformer.arch import STAEformer
    model = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=6, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt = 'checkpoints/STAEformer_5ch_mask/SAN_BERNARDINO_MASK_30_12_12/6c714b799b90a8654f0acfb9300bad42/STAEformer_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_staeformer_8ch_mask_aware(device):
    from baselines.STAEformer.arch import STAEformer
    model = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=6, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
    )
    ckpt = 'checkpoints/STAEformer_5ch_mask_aware/SAN_BERNARDINO_MASK_30_12_12/53ea25be2b661d1bc2e489768850fdef/STAEformer_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_stgcn_1ch(device):
    from baselines.STGCN.arch import STGCN
    adj_mx, _ = load_adj("datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl", "normlap")
    adj_mx = torch.Tensor(adj_mx[0])
    model = STGCN(
        Ks=3, Kt=3,
        blocks=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
        T=12, num_nodes=NUM_NODES,
        act_func='glu', graph_conv_type='cheb_graph_conv',
        adj_matrix=adj_mx, bias=True, droprate=0.5,
    )
    ckpt = 'checkpoints/STGCN/SAN_BERNARDINO_30_12_12/70d4d891bd58bc747849b30bd86f9774/STGCN_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


def load_stgcn_8ch_mask_aware(device):
    from baselines.STGCN.arch import STGCN
    adj_mx, _ = load_adj("datasets/SAN_BERNARDINO_MASK/adj_mx.pkl", "normlap")
    adj_mx = torch.Tensor(adj_mx[0])
    model = STGCN(
        Ks=3, Kt=3,
        blocks=[[8], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
        T=12, num_nodes=NUM_NODES,
        act_func='glu', graph_conv_type='cheb_graph_conv',
        adj_matrix=adj_mx, bias=True, droprate=0.5,
    )
    ckpt = 'checkpoints/STGCN_5ch_mask_aware/SAN_BERNARDINO_MASK_30_12_12/1b77f036e0b4c2135787d4bf52f513e1/STGCN_best_val_MAE.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model


class MICLDownstreamWrapper(torch.nn.Module):
    """Wrapper that chains MICL encoder -> STAEformer for eval."""
    def __init__(self, encoder, downstream, tod_idx=6, dow_idx=7):
        super().__init__()
        self.encoder = encoder
        self.downstream = downstream
        self.tod_idx = tod_idx
        self.dow_idx = dow_idx

    def forward(self, history_data, **kwargs):
        encoded = self.encoder.encode(history_data)  # [B, T, N, 32]
        # Reconstruct: [enc[0], tod, dow, enc[1:]] = 34 dims
        placeholder = encoded[..., 0:1]
        tod = history_data[..., self.tod_idx:self.tod_idx+1]
        dow = history_data[..., self.dow_idx:self.dow_idx+1]
        encoded_rest = encoded[..., 1:]
        history_enc = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)
        return self.downstream(history_data=history_enc, **kwargs)


def load_micl_downstream(device):
    import glob
    from baselines.STAEformer.arch import STAEformer
    from baselines.MICL.arch import MLPEncoder

    encoder = MLPEncoder(input_dim=8, d_model=32, hidden_dim=64)
    downstream = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=32, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
        tod_index=1, dow_index=2,
    )
    ckpt_pattern = 'checkpoints/MICL_downstream/SAN_BERNARDINO_MASK_30_12_12/*/STAEformer_best_val_MAE.pt'
    ckpt = sorted(glob.glob(ckpt_pattern))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    downstream.load_state_dict(state['model_state_dict'])
    if 'encoder_state_dict' in state:
        encoder.load_state_dict(state['encoder_state_dict'])
    model = MICLDownstreamWrapper(encoder, downstream)
    model.to(device).eval()
    return model


class MICLDownstreamV2Wrapper(torch.nn.Module):
    """Wrapper for MICL v2: encoder + pass-through mask channels -> STAEformer."""
    def __init__(self, encoder, downstream, tod_idx=6, dow_idx=7,
                 pass_through_indices=None):
        super().__init__()
        self.encoder = encoder
        self.downstream = downstream
        self.tod_idx = tod_idx
        self.dow_idx = dow_idx
        self.pass_through_indices = pass_through_indices or [3, 4, 5]

    def forward(self, history_data, **kwargs):
        encoded = self.encoder.encode(history_data)  # [B, T, N, 32]
        # Reconstruct: [enc[0], tod, dow, enc[1:31], mask_f, mask_o, mask_s] = 37 dims
        placeholder = encoded[..., 0:1]
        tod = history_data[..., self.tod_idx:self.tod_idx+1]
        dow = history_data[..., self.dow_idx:self.dow_idx+1]
        encoded_rest = encoded[..., 1:]
        pass_channels = history_data[..., self.pass_through_indices]
        history_enc = torch.cat([placeholder, tod, dow, encoded_rest, pass_channels], dim=-1)
        return self.downstream(history_data=history_enc, **kwargs)


def load_micl_downstream_v2(device):
    import glob
    from baselines.STAEformer.arch import STAEformer
    from baselines.MICL.arch import MLPEncoder

    encoder = MLPEncoder(input_dim=8, d_model=32, hidden_dim=64)
    downstream = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=37, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
        tod_index=1, dow_index=2,
    )
    ckpt_pattern = 'checkpoints/MICL_downstream_v2/SAN_BERNARDINO_MASK_30_12_12/*/STAEformer_best_val_MAE.pt'
    ckpt = sorted(glob.glob(ckpt_pattern))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    downstream.load_state_dict(state['model_state_dict'])
    if 'encoder_state_dict' in state:
        encoder.load_state_dict(state['encoder_state_dict'])
    model = MICLDownstreamV2Wrapper(encoder, downstream)
    model.to(device).eval()
    return model


def _load_micl_v1_arch(device, ckpt_pattern):
    """Shared loader for MICL v1-architecture models (v1, v3a, v3b, v3c)."""
    import glob
    from baselines.STAEformer.arch import STAEformer
    from baselines.MICL.arch import MLPEncoder

    encoder = MLPEncoder(input_dim=8, d_model=32, hidden_dim=64)
    downstream = STAEformer(
        num_nodes=NUM_NODES, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=32, output_dim=1, input_embedding_dim=24,
        tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
        adaptive_embedding_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1, use_mixed_proj=True,
        tod_index=1, dow_index=2,
    )
    ckpt = sorted(glob.glob(ckpt_pattern))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    downstream.load_state_dict(state['model_state_dict'])
    if 'encoder_state_dict' in state:
        encoder.load_state_dict(state['encoder_state_dict'])
    model = MICLDownstreamWrapper(encoder, downstream)
    model.to(device).eval()
    return model


def load_micl_downstream_v3a(device):
    return _load_micl_v1_arch(device, 'checkpoints/MICL_downstream_v3a/SAN_BERNARDINO_MASK_30_12_12/*/STAEformer_best_val_MAE.pt')

def load_micl_downstream_v3b(device):
    return _load_micl_v1_arch(device, 'checkpoints/MICL_downstream_v3b/SAN_BERNARDINO_MASK_30_12_12/*/STAEformer_best_val_MAE.pt')

def load_micl_downstream_v3c(device):
    return _load_micl_v1_arch(device, 'checkpoints/MICL_downstream_v3c/SAN_BERNARDINO_MASK_30_12_12/*/STAEformer_best_val_MAE.pt')


# ─── Model Registry ──────────────────────────────────────────────────
MODELS = {
    'staeformer_5ch': {
        'loader': load_staeformer_5ch,
        'dataset_name': 'xtraffic/SAN_BERNARDINO',
        'data_name': 'SAN_BERNARDINO',
        'forward_features': [0, 1, 2, 3, 4],
        'physical_channels': [0, 1, 2],   # flow, occ, speed
        'mask_channels': None,
    },
    'staeformer_5ch_maskloss': {
        'loader': load_staeformer_5ch_maskloss,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 6, 7],  # flow, occ, speed, tod, dow (skip mask ch)
        'physical_channels': [0, 1, 2],
        'mask_channels': None,  # model doesn't see masks -> can't distinguish corruption
    },
    'staeformer_8ch_mask': {
        'loader': load_staeformer_8ch_mask,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'staeformer_8ch_mask_aware': {
        'loader': load_staeformer_8ch_mask_aware,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'stgcn_1ch': {
        'loader': load_stgcn_1ch,
        'dataset_name': 'xtraffic/SAN_BERNARDINO',
        'data_name': 'SAN_BERNARDINO',
        'forward_features': [0],
        'physical_channels': [0],
        'mask_channels': None,
    },
    'stgcn_8ch_mask_aware': {
        'loader': load_stgcn_8ch_mask_aware,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'micl_downstream': {
        'loader': load_micl_downstream,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'micl_downstream_v2': {
        'loader': load_micl_downstream_v2,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'micl_v3a': {
        'loader': load_micl_downstream_v3a,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'micl_v3b': {
        'loader': load_micl_downstream_v3b,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
    'micl_v3c': {
        'loader': load_micl_downstream_v3c,
        'dataset_name': 'SAN_BERNARDINO_MASK',
        'data_name': 'SAN_BERNARDINO_MASK',
        'forward_features': [0, 1, 2, 3, 4, 5, 6, 7],
        'physical_channels': [0, 1, 2],
        'mask_channels': [3, 4, 5],
    },
}


# ─── Corruption Functions ────────────────────────────────────────────
def apply_node_death(inputs, corrupt_nodes, physical_channels, mask_channels):
    """Zero out all values for selected nodes across ALL timesteps.

    Simulates permanent sensor failure at test time.
    For 5ch: zeros physical channels (model can't distinguish from real zero).
    For 8ch: zeros physical + mask channels (model knows it's missing).
    """
    corrupted = inputs.clone()
    for ch in physical_channels:
        corrupted[:, :, corrupt_nodes, ch] = 0.0
    if mask_channels:
        for ch in mask_channels:
            corrupted[:, :, corrupt_nodes, ch] = 0.0
    return corrupted


def apply_intermittent(inputs, functional_nodes, rate, physical_channels,
                       mask_channels, rng):
    """Randomly zero out (timestep, node) entries with probability `rate`.

    Simulates increased random data loss at test time.
    """
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]
    n_func = len(functional_nodes)
    corrupt_mask = torch.tensor(
        rng.random((B, T, n_func)) < rate, dtype=torch.bool)
    for ch in physical_channels:
        vals = corrupted[:, :, functional_nodes, ch].clone()
        vals[corrupt_mask] = 0.0
        corrupted[:, :, functional_nodes, ch] = vals
    if mask_channels:
        for ch in mask_channels:
            vals = corrupted[:, :, functional_nodes, ch].clone()
            vals[corrupt_mask] = 0.0
            corrupted[:, :, functional_nodes, ch] = vals
    return corrupted


# ─── Inference ────────────────────────────────────────────────────────
def run_inference(model, inputs_norm, device, forward_features):
    """Run model forward pass, return predictions (B, 12, N) in normalized space."""
    history = inputs_norm[..., forward_features].to(device)
    with torch.no_grad():
        pred = model(history_data=history, future_data=None,
                     batch_seen=0, epoch=0, train=False)
    if isinstance(pred, dict):
        pred = pred['prediction']
    if pred.dim() == 4:
        pred = pred[..., 0]
    return pred.cpu()


# ─── Analysis ─────────────────────────────────────────────────────────
def compute_per_node_masked_mae(pred, targets, mask):
    """Compute per-node MAE on non-zero target entries only."""
    mae = np.zeros(pred.shape[2])
    for n in range(pred.shape[2]):
        m = mask[:, :, n]
        if m.sum() > 0:
            mae[n] = np.abs(pred[:, :, n][m] - targets[:, :, n][m]).mean()
    return mae


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model robustness to missing pattern shift')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to evaluate (default: all)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load functional node indices (same for both datasets)
    dead = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
    major_fail = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')
    functional = np.setdiff1d(np.arange(NUM_NODES),
                              np.union1d(dead, major_fail))
    print(f"Nodes: {NUM_NODES} total, {len(functional)} functional, "
          f"{len(dead)} dead, {len(major_fail)} major_fail")

    # Load target mask from 8ch dataset (observed/missing indicator)
    # This is used for ALL models - mask is a data property, not model property
    print("Loading target masks from 8ch dataset (mask_flow channel)...")
    rs_mask = get_regular_settings('SAN_BERNARDINO_MASK')
    mask_dataset = TimeSeriesForecastingDataset(
        dataset_name='SAN_BERNARDINO_MASK',
        train_val_test_ratio=rs_mask['TRAIN_VAL_TEST_RATIO'],
        mode='test', input_len=12, output_len=12,
        data_range=(0, 26280),
    )
    mask_loader = DataLoader(mask_dataset, batch_size=16, shuffle=False, num_workers=4)
    all_target_masks = []
    for batch in mask_loader:
        # mask_flow is channel 3 in 8ch: [flow, occ, speed, mask_f, mask_o, mask_s, tod, dow]
        all_target_masks.append(batch['target'][..., 3].float().numpy())
    target_mask_observed = np.concatenate(all_target_masks, axis=0)  # (N_test, 12, 893)
    print(f"  Target mask shape: {target_mask_observed.shape}")
    print(f"  Observed rate (overall): {target_mask_observed.mean():.4f}")
    print(f"  Observed rate (functional): {target_mask_observed[:, :, functional].mean():.4f}")

    # Pre-select corrupt nodes for node_death (shared across models for fairness)
    death_nodes = {}
    for shift_type, rate, label in SHIFT_CONFIGS:
        if shift_type == 'node_death':
            rng_select = np.random.RandomState(SEED + int(rate * 100))
            n_corrupt = int(len(functional) * rate)
            death_nodes[label] = np.sort(
                rng_select.choice(functional, n_corrupt, replace=False))

    models_to_run = args.models or list(MODELS.keys())
    all_results = {}

    for model_name in models_to_run:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        model_cfg = MODELS[model_name]
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"  Dataset: {model_cfg['dataset_name']}")
        print(f"  Forward features: {model_cfg['forward_features']}")
        print(f"  Mask channels: {model_cfg['mask_channels']}")
        print(f"{'='*80}")

        # Load model
        print("  Loading model...")
        model = model_cfg['loader'](device)

        # Load dataset and scaler
        dataset_name = model_cfg['dataset_name']
        data_name = model_cfg['data_name']
        rs = get_regular_settings(data_name)

        scaler = ZScoreScaler(
            dataset_name=dataset_name,
            train_ratio=rs['TRAIN_VAL_TEST_RATIO'][0],
            norm_each_channel=rs['NORM_EACH_CHANNEL'],
            rescale=rs['RESCALE'],
        )
        dataset = TimeSeriesForecastingDataset(
            dataset_name=dataset_name,
            train_val_test_ratio=rs['TRAIN_VAL_TEST_RATIO'],
            mode='test', input_len=12, output_len=12,
            data_range=(0, 26280),
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

        flow_mean = scaler.mean.float()
        flow_std = scaler.std.float()

        # ── Clean baseline ──
        print("  Computing clean predictions...")
        all_pred, all_targets = [], []
        for batch in loader:
            inputs_raw = batch['inputs'].float()
            target_raw = batch['target'].float()
            inputs_norm = inputs_raw.clone()
            inputs_norm[..., 0] = (inputs_norm[..., 0] - flow_mean) / flow_std
            pred_norm = run_inference(
                model, inputs_norm, device, model_cfg['forward_features'])
            all_pred.append((pred_norm * flow_std + flow_mean).numpy())
            all_targets.append(target_raw[..., 0].numpy())

        pred_clean = np.concatenate(all_pred, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Two evaluation masks
        mask_nonzero = targets > 0                          # legacy: excludes all zero targets
        mask_observed = target_mask_observed == 1           # correct: excludes only truly missing

        clean_mae_nz = compute_per_node_masked_mae(pred_clean, targets, mask_nonzero)
        clean_mae_ob = compute_per_node_masked_mae(pred_clean, targets, mask_observed)
        clean_func_mae_nz = float(clean_mae_nz[functional].mean())
        clean_func_mae_ob = float(clean_mae_ob[functional].mean())
        print(f"  Clean MAE (functional, target>0):      {clean_func_mae_nz:.4f}")
        print(f"  Clean MAE (functional, target_mask==1): {clean_func_mae_ob:.4f}")

        # ── Shift experiments ──
        model_results = {
            'model': model_name,
            'dataset': dataset_name,
            'clean_mae_functional_nonzero': clean_func_mae_nz,
            'clean_mae_functional_observed': clean_func_mae_ob,
            'n_functional': int(len(functional)),
            'configs': {},
        }

        for shift_type, rate, label in SHIFT_CONFIGS:
            print(f"  [{label}] ", end='', flush=True)

            if shift_type == 'node_death':
                corrupt_nodes = death_nodes[label]
                healthy = np.setdiff1d(functional, corrupt_nodes)

                all_pred_corrupted = []
                for batch in loader:
                    inputs_raw = batch['inputs'].float()
                    inputs_corrupted = apply_node_death(
                        inputs_raw, corrupt_nodes,
                        model_cfg['physical_channels'],
                        model_cfg['mask_channels'])
                    inputs_corrupted[..., 0] = \
                        (inputs_corrupted[..., 0] - flow_mean) / flow_std
                    pred_norm = run_inference(
                        model, inputs_corrupted, device,
                        model_cfg['forward_features'])
                    all_pred_corrupted.append(
                        (pred_norm * flow_std + flow_mean).numpy())
                pred_corrupted = np.concatenate(all_pred_corrupted, axis=0)

                corrupted_mae_nz = compute_per_node_masked_mae(
                    pred_corrupted, targets, mask_nonzero)
                corrupted_mae_ob = compute_per_node_masked_mae(
                    pred_corrupted, targets, mask_observed)

                # nonzero metric (legacy)
                h_clean_nz = float(clean_mae_nz[healthy].mean())
                h_corr_nz = float(corrupted_mae_nz[healthy].mean())
                deg_nz = h_corr_nz - h_clean_nz

                # observed metric (correct)
                h_clean_ob = float(clean_mae_ob[healthy].mean())
                h_corr_ob = float(corrupted_mae_ob[healthy].mean())
                deg_ob = h_corr_ob - h_clean_ob

                result = {
                    'shift_type': shift_type,
                    'rate': rate,
                    'n_corrupt': int(len(corrupt_nodes)),
                    'n_healthy': int(len(healthy)),
                    # observed metric (primary)
                    'healthy_clean_mae': h_clean_ob,
                    'healthy_corrupted_mae': h_corr_ob,
                    'degradation': deg_ob,
                    'pct_degradation': (deg_ob / h_clean_ob * 100
                                       if h_clean_ob > 0 else 0),
                    # nonzero metric (legacy comparison)
                    'healthy_clean_mae_nonzero': h_clean_nz,
                    'healthy_corrupted_mae_nonzero': h_corr_nz,
                    'degradation_nonzero': deg_nz,
                    'pct_degradation_nonzero': (deg_nz / h_clean_nz * 100
                                                if h_clean_nz > 0 else 0),
                }

            elif shift_type == 'intermittent':
                rng_corrupt = np.random.RandomState(
                    SEED + 2000 + int(rate * 100))

                all_pred_corrupted = []
                for batch in loader:
                    inputs_raw = batch['inputs'].float()
                    inputs_corrupted = apply_intermittent(
                        inputs_raw, functional, rate,
                        model_cfg['physical_channels'],
                        model_cfg['mask_channels'],
                        rng_corrupt)
                    inputs_corrupted[..., 0] = \
                        (inputs_corrupted[..., 0] - flow_mean) / flow_std
                    pred_norm = run_inference(
                        model, inputs_corrupted, device,
                        model_cfg['forward_features'])
                    all_pred_corrupted.append(
                        (pred_norm * flow_std + flow_mean).numpy())
                pred_corrupted = np.concatenate(all_pred_corrupted, axis=0)

                corrupted_mae_nz = compute_per_node_masked_mae(
                    pred_corrupted, targets, mask_nonzero)
                corrupted_mae_ob = compute_per_node_masked_mae(
                    pred_corrupted, targets, mask_observed)

                # nonzero metric (legacy)
                f_clean_nz = float(clean_mae_nz[functional].mean())
                f_corr_nz = float(corrupted_mae_nz[functional].mean())
                deg_nz = f_corr_nz - f_clean_nz

                # observed metric (correct)
                f_clean_ob = float(clean_mae_ob[functional].mean())
                f_corr_ob = float(corrupted_mae_ob[functional].mean())
                deg_ob = f_corr_ob - f_clean_ob

                result = {
                    'shift_type': shift_type,
                    'rate': rate,
                    'n_healthy': int(len(functional)),
                    # observed metric (primary)
                    'healthy_clean_mae': f_clean_ob,
                    'healthy_corrupted_mae': f_corr_ob,
                    'degradation': deg_ob,
                    'pct_degradation': (deg_ob / f_clean_ob * 100
                                       if f_clean_ob > 0 else 0),
                    # nonzero metric (legacy comparison)
                    'healthy_clean_mae_nonzero': f_clean_nz,
                    'healthy_corrupted_mae_nonzero': f_corr_nz,
                    'degradation_nonzero': deg_nz,
                    'pct_degradation_nonzero': (deg_nz / f_clean_nz * 100
                                                if f_clean_nz > 0 else 0),
                }

            model_results['configs'][label] = result
            print(f"observed: {result['healthy_clean_mae']:.3f} -> "
                  f"{result['healthy_corrupted_mae']:.3f} "
                  f"(d={result['degradation']:+.3f}, "
                  f"{result['pct_degradation']:+.1f}%) | "
                  f"nonzero: {result['healthy_clean_mae_nonzero']:.3f} -> "
                  f"{result['healthy_corrupted_mae_nonzero']:.3f} "
                  f"(d={result['degradation_nonzero']:+.3f}, "
                  f"{result['pct_degradation_nonzero']:+.1f}%)")

        all_results[model_name] = model_results

        # Save per-model clean MAE (both metrics)
        np.save(os.path.join(OUTPUT_DIR, f'{model_name}_clean_mae_nonzero.npy'),
                clean_mae_nz)
        np.save(os.path.join(OUTPUT_DIR, f'{model_name}_clean_mae_observed.npy'),
                clean_mae_ob)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Save all results (merge with existing) ──
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print Summary Tables ──
    print_summary_table(all_results)

    print(f"\nResults saved to {OUTPUT_DIR}/")


def print_summary_table(all_results):
    """Print comparison tables across all models."""
    models = list(all_results.keys())
    if not models:
        return

    col_w = 25

    for metric_key, metric_name, clean_key, deg_key, pct_key, mae_key in [
        ('', 'target_mask==1 (observed)', 'clean_mae_functional_observed',
         'degradation', 'pct_degradation', 'healthy_corrupted_mae'),
        ('_nonzero', 'target>0 (legacy)', 'clean_mae_functional_nonzero',
         'degradation_nonzero', 'pct_degradation_nonzero', 'healthy_corrupted_mae_nonzero'),
    ]:
        print(f"\n{'='*130}")
        print(f"MISSING PATTERN SHIFT [{metric_name}]: Degradation (MAE change on healthy nodes)")
        print(f"{'='*130}")

        header = f"{'Config':<18}"
        for m in models:
            header += f" | {m:>{col_w}}"
        print(header)
        print("-" * len(header))

        row = f"{'Clean MAE':<18}"
        for m in models:
            row += f" | {all_results[m][clean_key]:>{col_w}.3f}"
        print(row)
        print("-" * len(header))

        for shift_type, rate, label in SHIFT_CONFIGS:
            row = f"{label:<18}"
            for m in models:
                if label in all_results[m]['configs']:
                    d = all_results[m]['configs'][label][deg_key]
                    pct = all_results[m]['configs'][label][pct_key]
                    cell = f"{d:+.3f} ({pct:+.1f}%)"
                    row += f" | {cell:>{col_w}}"
                else:
                    row += f" | {'N/A':>{col_w}}"
            print(row)

        print(f"{'='*130}")


if __name__ == '__main__':
    main()
