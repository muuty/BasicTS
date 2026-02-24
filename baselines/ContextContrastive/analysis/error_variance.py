"""
Error Variance Analysis: Compare baseline vs pre-trained encoder

Metrics:
- Per-node MAE mean and std
- Worst-case MAE (top 5%, 10%)
- CV (Coefficient of Variation)
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__ + '/../../../..'))

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import load_adj

from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.arch.context_contrastive_model import TemporalEncoder


def get_predictions_baseline(device='cuda:0'):
    """Get predictions from baseline STAEformer."""
    # Dataset
    dataset = TimeSeriesForecastingDataset(
        dataset_name='PEMS08',
        train_val_test_ratio=[0.6, 0.2, 0.2],
        input_len=12,
        output_len=12,
        mode='test'
    )

    # Scaler
    scaler = ZScoreScaler(
        dataset_name='PEMS08',
        train_ratio=0.6,
        norm_each_channel=False,
        rescale=True,
    )

    # Model
    adj_mx, _ = load_adj("datasets/PEMS08/adj_mx.pkl", "normlap")
    adj_mx = torch.Tensor(adj_mx[0])

    model = STAEformer(
        num_nodes=170,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=24,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        use_mixed_proj=True,
    ).to(device)

    # Find checkpoint
    ckpt_dir = 'checkpoints/STAEformer/PEMS08_30_12_12'
    ckpt_path = None
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if 'best_val_MAE' in f:
                ckpt_path = os.path.join(root, f)
                break

    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Predict
    return _predict(model, dataset, scaler, device, encoder=None, include_tod_dow=False)


def get_predictions_pretrained(device='cuda:0'):
    """Get predictions from pre-trained encoder + STAEformer."""
    # Dataset
    dataset = TimeSeriesForecastingDataset(
        dataset_name='PEMS08',
        train_val_test_ratio=[0.6, 0.2, 0.2],
        input_len=12,
        output_len=12,
        mode='test'
    )

    # Scaler
    scaler = ZScoreScaler(
        dataset_name='PEMS08',
        train_ratio=0.6,
        norm_each_channel=False,
        rescale=True,
    )

    # Model (input_dim = 64 + 2 = 66 for [placeholder, tod, dow, encoded_rest])
    D_MODEL = 64
    model = STAEformer(
        num_nodes=170,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=D_MODEL + 2,  # 66
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=24,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        use_mixed_proj=True,
    ).to(device)

    # Find checkpoint
    ckpt_dir = 'checkpoints/ContextContrastive_STAEformer/PEMS08_30_12_12'
    ckpt_path = None
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if 'best_val_MAE' in f:
                ckpt_path = os.path.join(root, f)
                break

    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load encoder
    encoder = TemporalEncoder(
        c_in=3,
        d_model=D_MODEL,
        num_layers=2,
        nhead=4,
        dropout=0.1,
    ).to(device)

    enc_ckpt_path = 'checkpoints/ContextContrastive_pretrain/PEMS08_50_12_12/4e08bf9303d45ae6dc4c084d44776987/ContextContrastiveModel_best_val_MAE.pt'
    enc_ckpt = torch.load(enc_ckpt_path, map_location=device)
    enc_state = enc_ckpt.get('model_state_dict', enc_ckpt)
    enc_state_filtered = {k.replace('temporal_encoder.', ''): v
                         for k, v in enc_state.items()
                         if k.startswith('temporal_encoder.')}
    encoder.load_state_dict(enc_state_filtered)
    encoder.eval()
    print(f"Loaded pre-trained encoder from {enc_ckpt_path}")

    # Predict
    return _predict(model, dataset, scaler, device, encoder=encoder, include_tod_dow=True)


def _predict(model, dataset, scaler, device, encoder=None, include_tod_dow=False):
    """Run prediction loop."""
    all_preds = []
    all_targets = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = batch['inputs'].to(device)  # [B, L, N, C]
            target = batch['target'].to(device)  # [B, L, N, C]

            # Scale inputs
            inputs_scaled = scaler.transform(inputs)

            # Select features [0, 1, 2]
            history = inputs_scaled[..., [0, 1, 2]]

            # Apply encoder if exists
            if encoder is not None:
                encoded = encoder(history)
                if include_tod_dow:
                    # Structure: [placeholder, tod, dow, encoded_rest]
                    placeholder = encoded[..., 0:1]
                    tod = history[..., 1:2]
                    dow = history[..., 2:3]
                    encoded_rest = encoded[..., 1:]
                    history = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)
                else:
                    history = encoded

            # Forward
            pred = model(
                history_data=history,
                future_data=None,
                batch_seen=0,
                epoch=0,
                train=False
            )

            if isinstance(pred, dict):
                pred = pred['prediction']

            # Inverse scale
            pred = scaler.inverse_transform(pred)

            # Target feature [0]
            target_val = target[..., [0]]

            all_preds.append(pred.cpu())
            all_targets.append(target_val.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return preds.numpy(), targets.numpy()


def compute_metrics(preds: np.ndarray, targets: np.ndarray, null_val: float = 0.0):
    """Compute error metrics."""
    # Squeeze last dim
    preds = preds.squeeze(-1)  # [N_samples, L, N_nodes]
    targets = targets.squeeze(-1)

    # Mask for null values
    mask = (targets != null_val)

    # Per-sample, per-node absolute error
    abs_error = np.abs(preds - targets)
    abs_error = np.where(mask, abs_error, np.nan)

    # Per-node MAE (average over samples and time)
    node_mae = np.nanmean(abs_error, axis=(0, 1))  # [N_nodes]

    # Overall MAE
    overall_mae = np.nanmean(abs_error)

    # Std of per-node MAE
    node_mae_std = np.std(node_mae)

    # CV (Coefficient of Variation)
    node_mae_cv = node_mae_std / np.mean(node_mae)

    # Worst-case MAE (top 5%, 10%)
    sorted_node_mae = np.sort(node_mae)[::-1]
    n_nodes = len(node_mae)
    worst_5pct_mae = np.mean(sorted_node_mae[:max(1, int(n_nodes * 0.05))])
    worst_10pct_mae = np.mean(sorted_node_mae[:max(1, int(n_nodes * 0.10))])

    # Per-node error std
    node_error_std = np.nanstd(abs_error, axis=(0, 1))
    avg_node_error_std = np.mean(node_error_std)

    return {
        'overall_mae': overall_mae,
        'node_mae_mean': np.mean(node_mae),
        'node_mae_std': node_mae_std,
        'node_mae_cv': node_mae_cv,
        'worst_5pct_mae': worst_5pct_mae,
        'worst_10pct_mae': worst_10pct_mae,
        'avg_node_error_std': avg_node_error_std,
        'node_mae': node_mae,
    }


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Error Variance Analysis")
    print("=" * 60)

    # Baseline
    print("\n[1] Baseline STAEformer")
    print("-" * 40)
    baseline_preds, baseline_targets = get_predictions_baseline(device)
    baseline_metrics = compute_metrics(baseline_preds, baseline_targets)

    # Pre-trained encoder
    print("\n[2] Pre-trained Encoder + STAEformer")
    print("-" * 40)
    pretrained_preds, pretrained_targets = get_predictions_pretrained(device)
    pretrained_metrics = compute_metrics(pretrained_preds, pretrained_targets)

    # Print comparison
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    metrics_to_compare = [
        ('overall_mae', 'Overall MAE'),
        ('node_mae_std', 'Node MAE Std'),
        ('node_mae_cv', 'Node MAE CV'),
        ('worst_5pct_mae', 'Worst 5% MAE'),
        ('worst_10pct_mae', 'Worst 10% MAE'),
        ('avg_node_error_std', 'Avg Node Error Std'),
    ]

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Pretrained':>12} {'Change':>12}")
    print("-" * 65)

    for key, name in metrics_to_compare:
        b_val = baseline_metrics[key]
        p_val = pretrained_metrics[key]
        change = (p_val - b_val) / b_val * 100
        indicator = "" if change < 0 else "+"
        print(f"{name:<25} {b_val:>12.4f} {p_val:>12.4f} {indicator}{change:>10.2f}%")

    # Node-wise improvement analysis
    baseline_node_mae = baseline_metrics['node_mae']
    pretrained_node_mae = pretrained_metrics['node_mae']

    improved_nodes = np.sum(pretrained_node_mae < baseline_node_mae)
    total_nodes = len(baseline_node_mae)

    print(f"\n{'Improved Nodes':<25} {improved_nodes}/{total_nodes} ({improved_nodes/total_nodes*100:.1f}%)")

    # Improvement distribution
    improvement = (baseline_node_mae - pretrained_node_mae) / baseline_node_mae * 100
    print(f"\nPer-Node Improvement Distribution:")
    print(f"  Min:  {improvement.min():>7.2f}%")
    print(f"  25%:  {np.percentile(improvement, 25):>7.2f}%")
    print(f"  50%:  {np.percentile(improvement, 50):>7.2f}%")
    print(f"  75%:  {np.percentile(improvement, 75):>7.2f}%")
    print(f"  Max:  {improvement.max():>7.2f}%")

    # Deeper analysis: why did std increase?
    print("\n" + "=" * 60)
    print("Deeper Analysis: Node MAE Distribution")
    print("=" * 60)

    # Quartile analysis
    print(f"\nBaseline Node MAE Distribution:")
    print(f"  Min:  {baseline_node_mae.min():>7.2f}")
    print(f"  25%:  {np.percentile(baseline_node_mae, 25):>7.2f}")
    print(f"  50%:  {np.percentile(baseline_node_mae, 50):>7.2f}")
    print(f"  75%:  {np.percentile(baseline_node_mae, 75):>7.2f}")
    print(f"  Max:  {baseline_node_mae.max():>7.2f}")

    print(f"\nPretrained Node MAE Distribution:")
    print(f"  Min:  {pretrained_node_mae.min():>7.2f}")
    print(f"  25%:  {np.percentile(pretrained_node_mae, 25):>7.2f}")
    print(f"  50%:  {np.percentile(pretrained_node_mae, 50):>7.2f}")
    print(f"  75%:  {np.percentile(pretrained_node_mae, 75):>7.2f}")
    print(f"  Max:  {pretrained_node_mae.max():>7.2f}")

    # Which nodes got worse?
    worse_mask = pretrained_node_mae > baseline_node_mae
    worse_nodes = np.where(worse_mask)[0]
    print(f"\nNodes that got worse ({len(worse_nodes)}):")
    for node_id in worse_nodes:
        b_mae = baseline_node_mae[node_id]
        p_mae = pretrained_node_mae[node_id]
        print(f"  Node {node_id:3d}: {b_mae:.2f} -> {p_mae:.2f} ({(p_mae-b_mae)/b_mae*100:+.1f}%)")

    # Best vs worst nodes: did the gap change?
    baseline_sorted = np.sort(baseline_node_mae)
    pretrained_sorted = np.sort(pretrained_node_mae)

    print(f"\nBest 10 nodes (avg MAE):")
    print(f"  Baseline:   {baseline_sorted[:10].mean():.2f}")
    print(f"  Pretrained: {pretrained_sorted[:10].mean():.2f}")

    print(f"\nWorst 10 nodes (avg MAE):")
    print(f"  Baseline:   {baseline_sorted[-10:].mean():.2f}")
    print(f"  Pretrained: {pretrained_sorted[-10:].mean():.2f}")

    print(f"\nGap (Worst10 - Best10):")
    print(f"  Baseline:   {baseline_sorted[-10:].mean() - baseline_sorted[:10].mean():.2f}")
    print(f"  Pretrained: {pretrained_sorted[-10:].mean() - pretrained_sorted[:10].mean():.2f}")


if __name__ == '__main__':
    main()
