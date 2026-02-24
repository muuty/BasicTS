import torch
import numpy as np
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
import sys
import csv
import ot
from scipy.spatial.distance import cdist
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimal_transport import generate_projections, project_data
from basicts.data import TimeSeriesForecastingDataset
from gumbel import GumbelSigmoidCoresetOptimizer, quantile_resample
from k_medoids import kmedoids_selection

# ===== Dataset & Feature Extraction =====

def get_dataset(dataset_name: str, input_len: int, output_len: int, train_val_test_ratio: tuple, mode: str):
    return TimeSeriesForecastingDataset(
        dataset_name=dataset_name, 
        input_len=input_len, 
        output_len=output_len, 
        train_val_test_ratio=train_val_test_ratio, 
        data_range=(0, 21492),
        mode=mode
    )


def extract_features(dataset, indices, model_config, target_dim=128, seed=42):
    mean = np.mean(dataset.data, axis=(0, 1), keepdims=True)
    std = np.std(dataset.data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    def transform(x):
        return (x - mean) / std

    features = []
    for idx in tqdm(indices, desc="Extracting features"):
        sample = dataset[idx]
        
        inputs = transform(sample['inputs'])[:, :, model_config.FORWARD_FEATURES]
        target = transform(sample['target'])[:, :, model_config.TARGET_FEATURES]
        raw_feat = np.concatenate([inputs.reshape(-1), target.reshape(-1)])
        
        features.append(raw_feat)

    return np.array(features, dtype=np.float32)


def main():
    import os
    import sys
    from easydict import EasyDict
    from datetime import datetime
    
    sys.path.append(os.path.abspath(__file__ + '/../..'))
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from basicts.data import TimeSeriesForecastingDataset
    

    print(f"\n{'='*60}")
    print("Fast ARO + Sliced Wasserstein Coreset Selection")
    print(f"{'='*60}\n")
    
    # Config
    dataset_name = 'xtraffic/SAN_BERNARDINO'
    model_config = EasyDict()
    model_config.FORWARD_FEATURES = [0, 1, 2]
    model_config.TARGET_FEATURES = [0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("Loading datasets...")
    train_dataset = get_dataset(dataset_name, 12, 12, (0.6, 0.2, 0.2), 'train')
    val_dataset = get_dataset(dataset_name, 12, 12, (0.6, 0.2, 0.2), 'valid')
    test_dataset = get_dataset(dataset_name, 12, 12, (0.6, 0.2, 0.2), 'test')
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Extract features
    print("\nExtracting features...")
    train_features = extract_features(train_dataset, range(len(train_dataset)), model_config)
    val_features = extract_features(val_dataset, range(len(val_dataset)), model_config)
    test_features = extract_features(test_dataset, range(len(test_dataset)), model_config)
    print(f"Train features: {train_features.shape}, Val features: {val_features.shape}, Test features: {test_features.shape}")
    
    
    # PCA 적용
    print("\nApplying PCA...")
    pca_dims = [200]
    ratios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_train = len(train_dataset)
    
    for n_components in pca_dims:
        pca = PCA(n_components=n_components, random_state=42)
        train_pca = pca.fit_transform(train_features)
        val_pca = pca.transform(val_features)
        test_pca = pca.transform(test_features)
        # Val weights (고정)
        b = np.ones(len(val_pca)) / len(val_pca)
        b_test = np.ones(len(test_pca)) / len(test_pca)
        
        M_global = cdist(train_pca, val_pca, metric='euclidean')
        M_global_test = cdist(train_pca, test_pca, metric='euclidean')
        GLOBAL_MAX = M_global.max()
        GLOBAL_MAX_TEST = M_global_test.max()
        del M_global, M_global_test

        for ratio in ratios:
            k = int(n_train * ratio)
            
            w_distances_val = []
            w_distances_test = []
            for seed in range(1):
                # ===== 1. Selection 단계 (Semi-relaxed OT) =====
                n_source = len(train_pca)
                n_target = len(val_pca)
                
                # Source: 각 점의 용량은 1/k (총 질량 = N/k > 1.0)
                a = np.ones(n_source) / k  
                
                # Target: 원래 점들 (총 질량 1.0) + 쓰레기통 (남는 질량)
                total_source_mass = np.sum(a)
                dummy_mass = total_source_mass - 1.0 # 버려야 할 질량
                
                # Target 분포 확장: [원래 Target, 쓰레기통]
                b_real = np.ones(n_target) / n_target
                b_extended = np.append(b_real, dummy_mass)
                
                # Cost Matrix 확장
                M_val = cdist(train_pca, val_pca, metric='euclidean')
                M_norm_val = M_val / GLOBAL_MAX
                M_test = cdist(train_pca, test_pca, metric='euclidean')
                M_norm_test = M_test / GLOBAL_MAX_TEST
                


                # 쓰레기통으로 가는 비용은 0 (버리는 건 공짜)
                # (N, M) -> (N, M+1)
                zeros_column = np.zeros((n_source, 1))
                M_extended_val = np.hstack([M_norm_val, zeros_column])
                M_extended_test = np.hstack([M_norm_test, zeros_column])
                
                # 일반 EMD 풀기 (이제 총 질량이 양쪽 다 total_source_mass로 같음)
                # numItermax를 넉넉히 줍니다.
                gamma_extended_val = ot.emd(a, b_extended, M_extended_val, numItermax=2000000)
                gamma_extended_test = ot.emd(a, b_extended, M_extended_test, numItermax=2000000)
                
                # 쓰레기통(마지막 컬럼)을 제외한 나머지 부분이 진짜 Transport Plan
                gamma_val = gamma_extended_val[:, :-1]
                gamma_test = gamma_extended_test[:, :-1]
                
                # 선택된 Indices 추출 (위와 동일)
                source_mass_val = gamma_val.sum(axis=1)
                source_mass_test = gamma_test.sum(axis=1)
                
                # 부동소수점 오차 고려하여 k개 선택
                selected_indices_val = np.argsort(source_mass_val)[-k:]
                selected_indices_test = np.argsort(source_mass_test)[-k:]
                selected_val = train_pca[selected_indices_val]
                selected_test = train_pca[selected_indices_test]
                
                # ===== 2. Evaluation 단계 (Exact EMD) =====
                # 선택된 Subset만 가지고 다시 정밀 측정
                M_selected_val = cdist(selected_val, val_pca, metric='euclidean')
                M_selected_test = cdist(selected_test, test_pca, metric='euclidean')
                
                # 여기서는 Evaluation이므로 Global Scale 유지
                M_selected_norm_val = M_selected_val / GLOBAL_MAX
                M_selected_norm_test = M_selected_test / GLOBAL_MAX_TEST

                a_selected_val = np.ones(len(selected_val)) / len(selected_val)
                a_selected_test = np.ones(len(selected_test)) / len(selected_test)
                
                # 최종 거리 측정 (Global Max 곱해서 원래 단위 복원)
                w_dist_val = ot.emd2(a_selected_val, b_real, M_selected_norm_val, numItermax=500000000) * GLOBAL_MAX
                w_dist_test = ot.emd2(a_selected_test, b_test, M_selected_norm_test, numItermax=500000000) * GLOBAL_MAX_TEST
                
                w_distances_val.append(w_dist_val)
                w_distances_test.append(w_dist_test)


            print(f"{ratio:<8.2f} {np.mean(w_distances_val):.6f} ± {np.std(w_distances_val):.6f} {np.mean(w_distances_test):.6f} ± {np.std(w_distances_test):.6f}")


if __name__ == "__main__":
    main()