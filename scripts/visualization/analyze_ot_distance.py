"""
OT Distance 분석 스크립트.

기존 실험 결과에서 coreset indices를 읽어 OT distance를 계산하고,
모델 성능과의 상관관계를 분석.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))

from coreset.ot_distance import OTDistanceCalculator


def extract_features_from_dataset(dataset_path: str, data_range: tuple = None):
    """
    데이터셋에서 features 추출.

    Args:
        dataset_path: 데이터셋 경로 (e.g., 'datasets/xtraffic/SAN_BERNARDINO')
        data_range: (start, end) tuple for data range

    Returns:
        features: (N, D) numpy array
    """
    import numpy as np

    # Load data
    data_path = os.path.join(dataset_path, 'data.dat')
    desc_path = os.path.join(dataset_path, 'desc.json')

    with open(desc_path, 'r') as f:
        desc = json.load(f)

    shape = desc['shape']
    data = np.memmap(data_path, dtype=np.float32, mode='r', shape=tuple(shape))

    # Apply data range
    if data_range:
        data = data[data_range[0]:data_range[1]]

    # Normalize
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    data_normalized = (data - mean) / std

    # Create sliding window samples (simplified)
    input_len = 12
    output_len = 12
    num_samples = len(data_normalized) - input_len - output_len + 1

    # Train ratio
    train_ratio = 0.6
    train_samples = int(num_samples * train_ratio)

    features = []
    for i in range(train_samples):
        inputs = data_normalized[i:i + input_len]
        target = data_normalized[i + input_len:i + input_len + output_len]

        # Use only flow feature (index 0) for simplicity
        feat = np.concatenate([
            inputs[:, :, 0].reshape(-1),
            target[:, :, 0].reshape(-1)
        ])
        features.append(feat)

    return np.array(features, dtype=np.float32)


def analyze_experiment_results(
    checkpoints_dir: str,
    dataset_path: str,
    output_csv: str = None,
    pca_dim: int = 50,
    epsilon: float = 0.1
):
    """
    실험 결과 분석.

    Args:
        checkpoints_dir: checkpoints 디렉토리 (e.g., 'checkpoints/coreset')
        dataset_path: 데이터셋 경로
        output_csv: 결과 저장 CSV 경로
        pca_dim: PCA 차원
        epsilon: Sinkhorn epsilon
    """
    results = []

    # Extract full dataset features
    print("Extracting features from full dataset...")
    full_features = extract_features_from_dataset(dataset_path)
    print(f"Full dataset: {full_features.shape}")

    # Initialize OT calculator
    ot_calculator = OTDistanceCalculator(pca_dim=pca_dim, epsilon=epsilon)
    ot_calculator.fit_pca(full_features)

    # Walk through checkpoints
    for root, dirs, files in os.walk(checkpoints_dir):
        if 'coreset-selection.json' in files and 'test_metrics.json' in files:
            # Load coreset indices
            with open(os.path.join(root, 'coreset-selection.json'), 'r') as f:
                selected_indices = json.load(f)

            # Load metrics
            with open(os.path.join(root, 'test_metrics.json'), 'r') as f:
                metrics = json.load(f)

            # Extract experiment info from path
            path_parts = root.split('/')
            # e.g., checkpoints/coreset/STGCN/xtraffic/SAN_BERNARDINO_50_12_12/1
            model = path_parts[-4] if len(path_parts) >= 4 else 'unknown'
            dataset = path_parts[-2] if len(path_parts) >= 2 else 'unknown'

            # Parse ratio from dataset name (e.g., SAN_BERNARDINO_50_12_12 -> 0.5)
            try:
                ratio_str = dataset.split('_')[-3]
                ratio = int(ratio_str) / 100
            except:
                ratio = 1.0

            # Compute OT distance
            coreset_features = full_features[selected_indices]
            ot_distance = ot_calculator.compute_distance(
                coreset_features, full_features, subsample_full=5000
            )

            result = {
                'path': root,
                'model': model,
                'dataset': dataset,
                'ratio': ratio,
                'n_selected': len(selected_indices),
                'ot_distance': ot_distance,
                'MAE': metrics.get('MAE', {}).get('all', None),
                'RMSE': metrics.get('RMSE', {}).get('all', None),
                'MAPE': metrics.get('MAPE', {}).get('all', None),
            }
            results.append(result)

            print(f"{model} | ratio={ratio:.1f} | OT={ot_distance:.4f} | MAE={result['MAE']:.4f}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Compute correlations
    if len(df) > 2:
        print("\n" + "=" * 50)
        print("Correlation Analysis: OT Distance vs Performance")
        print("=" * 50)

        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in df.columns and df[metric].notna().sum() > 2:
                corr, pval = stats.pearsonr(
                    df['ot_distance'].dropna(),
                    df[metric].dropna()
                )
                print(f"OT vs {metric}: r={corr:.4f}, p={pval:.4f}")

    # Save results
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze OT distance vs model performance')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/coreset',
                        help='Checkpoints directory')
    parser.add_argument('--dataset', type=str, default='datasets/xtraffic/SAN_BERNARDINO',
                        help='Dataset path')
    parser.add_argument('--output', type=str, default='experiments/result/ot_analysis.csv',
                        help='Output CSV path')
    parser.add_argument('--pca_dim', type=int, default=50, help='PCA dimension')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Sinkhorn epsilon')

    args = parser.parse_args()

    df = analyze_experiment_results(
        checkpoints_dir=args.checkpoints,
        dataset_path=args.dataset,
        output_csv=args.output,
        pca_dim=args.pca_dim,
        epsilon=args.epsilon
    )

    print("\n" + "=" * 50)
    print("Summary by Ratio")
    print("=" * 50)
    print(df.groupby('ratio')[['ot_distance', 'MAE']].mean())


if __name__ == '__main__':
    main()
