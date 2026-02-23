"""
각 ratio에 대해 k_medoids selection을 실행하고 OT distance를 계산.
기존 실험 결과의 성능과 비교 분석.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from basicts.data import TimeSeriesForecastingDataset
from coreset.k_medoids import KMedoidsSelection
from coreset.ot_distance import OTDistanceCalculator
from easydict import EasyDict


def compute_ot_distances_for_dataset(
    dataset_name: str = 'xtraffic/SAN_BERNARDINO',
    ratios: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    seed: int = 42,
    pca_dim: int = 50,
    epsilon: float = 0.1,
    data_range: tuple = (0, 24192)  # 3 months
):
    """
    각 ratio에 대해 k_medoids selection 실행 후 OT distance 계산.
    """
    print(f"Dataset: {dataset_name}")
    print(f"Data range: {data_range}")
    print(f"Ratios: {ratios}")
    print(f"Seed: {seed}")
    print("=" * 60)

    # Load dataset
    dataset = TimeSeriesForecastingDataset(
        dataset_name=dataset_name,
        train_val_test_ratio=[0.6, 0.2, 0.2],
        input_len=12,
        output_len=12,
        mode='train',
        data_range=data_range
    )

    print(f"Dataset size: {len(dataset)}")

    # Model config (for feature extraction)
    model_config = EasyDict({
        'FORWARD_FEATURES': [0],  # flow only
        'TARGET_FEATURES': [0]
    })

    # Extract all features first
    print("\nExtracting features from full dataset...")
    dataset_size = len(dataset)
    mean = np.mean(dataset.data, axis=(0, 1), keepdims=True)
    std = np.std(dataset.data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    def transform(input_data):
        return (input_data - mean) / std

    features = []
    for i in range(dataset_size):
        sample = dataset[i]
        inputs = transform(sample['inputs'])[:, :, model_config.FORWARD_FEATURES]
        target = transform(sample['target'])[:, :, model_config.TARGET_FEATURES]
        feat = np.concatenate([inputs.reshape(-1), target.reshape(-1)])
        features.append(feat)

    features = np.array(features, dtype=np.float32)
    print(f"Features shape: {features.shape}")

    # Initialize OT calculator
    ot_calculator = OTDistanceCalculator(pca_dim=pca_dim, epsilon=epsilon)
    ot_calculator.fit_pca(features)

    # Compute OT distance for each ratio
    results = []

    for ratio in ratios:
        print(f"\n--- Ratio: {ratio} ---")

        if ratio == 1.0:
            # Full dataset
            selected_indices = list(range(dataset_size))
            ot_distance = 0.0  # Distance to itself
        else:
            # Run k_medoids selection
            selection = KMedoidsSelection(
                dataset=dataset,
                ratio=ratio,
                embedding_model=None,
                model_config=model_config,
                seed=seed
            )
            selected_indices = selection.select_indices()

            # Compute OT distance
            coreset_features = features[selected_indices]
            ot_distance = ot_calculator.compute_distance(
                coreset_features, features, subsample_full=5000
            )

        results.append({
            'ratio': ratio,
            'n_selected': len(selected_indices),
            'ot_distance': ot_distance
        })

        print(f"Selected: {len(selected_indices)}, OT Distance: {ot_distance:.4f}")

    return pd.DataFrame(results)


def merge_with_performance(ot_df: pd.DataFrame, performance_csv: str, model: str = 'STAEformer'):
    """
    OT distance 결과와 성능 결과 병합.
    """
    perf_df = pd.read_csv(performance_csv)

    # Filter by model
    perf_df = perf_df[perf_df['Model'] == model].copy()
    perf_df = perf_df.rename(columns={'Ratio': 'ratio'})

    # Merge
    merged = pd.merge(ot_df, perf_df, on='ratio', how='left')

    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='xtraffic/SAN_BERNARDINO')
    parser.add_argument('--model', type=str, default='STAEformer')
    parser.add_argument('--performance_csv', type=str,
                        default='experiments/result/coreset_san_bernardino_summary.csv')
    parser.add_argument('--output', type=str, default='experiments/result/ot_analysis.csv')
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Compute OT distances
    ot_df = compute_ot_distances_for_dataset(
        dataset_name=args.dataset,
        pca_dim=args.pca_dim,
        epsilon=args.epsilon,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("OT Distance Results:")
    print(ot_df.to_string(index=False))

    # Merge with performance
    if os.path.exists(args.performance_csv):
        merged_df = merge_with_performance(ot_df, args.performance_csv, args.model)

        print("\n" + "=" * 60)
        print(f"Merged Results ({args.model}):")
        print(merged_df[['ratio', 'n_selected', 'ot_distance', 'MAE_mean', 'MAE_std']].to_string(index=False))

        # Correlation analysis
        if len(merged_df) > 2 and 'MAE_mean' in merged_df.columns:
            from scipy import stats

            # Exclude ratio=1.0 for correlation (OT=0)
            analysis_df = merged_df[merged_df['ratio'] < 1.0]

            if len(analysis_df) > 2:
                corr, pval = stats.pearsonr(
                    analysis_df['ot_distance'],
                    analysis_df['MAE_mean']
                )
                print(f"\nCorrelation (OT vs MAE, excluding ratio=1.0):")
                print(f"  Pearson r = {corr:.4f}, p-value = {pval:.4f}")

        # Save
        merged_df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")
    else:
        ot_df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
