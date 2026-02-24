import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import csv
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimal_transport import generate_projections, project_data
from basicts.data import TimeSeriesForecastingDataset
from gumbel import GumbelSigmoidCoresetOptimizer, quantile_resample

@torch.no_grad()
def prepare_sliced_metric(
    train_features: np.ndarray,
    val_features: np.ndarray,
    n_projections: int,
    device: str = "cuda",
    seed: int = 42,
):
    """
    train_features, val_features로부터
    - train_proj: (N, P)
    - val_sorted: (M, P)  (projection 후 column-wise sort)
    를 생성한다.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, D = train_features.shape
    projections = generate_projections(D, n_projections, device)

    train_t = torch.from_numpy(train_features).float().to(device)
    val_t   = torch.from_numpy(val_features).float().to(device)

    train_proj = project_data(train_t, projections)  # (N, P)
    val_proj   = project_data(val_t, projections)    # (M, P)

    val_sorted, _ = torch.sort(val_proj, dim=0)      # (M, P)
    return train_proj, val_sorted   
@torch.no_grad()
def sw_uniform_exact_k(
    train_proj: torch.Tensor,    # (N, P) on device
    val_sorted: torch.Tensor,    # (M, P) on device, already sorted along dim=0
    indices: torch.Tensor,       # (k,) long on device
    n_common: int = 4096
) -> float:
    """
    exact-k subset을 '균일 가중치 empirical measure'로 보고,
    Sliced-Wasserstein(1D quantile distance 평균)을 계산한다.
    """
    src = train_proj[indices]                     # (k, P)
    src_sorted, _ = torch.sort(src, dim=0)        # (k, P)

    src_interp = quantile_resample(src_sorted, n_common)  # (n_common, P)
    val_interp = quantile_resample(val_sorted, n_common)  # (n_common, P)

    sw = torch.mean(torch.abs(src_interp - val_interp))
    return float(sw.item())

# ===== Dataset & Feature Extraction =====

def get_dataset(dataset_name: str, input_len: int, output_len: int, train_val_test_ratio: tuple, mode: str):
    return TimeSeriesForecastingDataset(
        dataset_name=dataset_name, 
        input_len=input_len, 
        output_len=output_len, 
        train_val_test_ratio=train_val_test_ratio, 
        mode=mode
    )


def extract_features(dataset, indices, model_config, target_dim=128, seed=42):
    """
    Gaussian Random Projection을 이용한 초고속 차원 축소
    """
    # 1. 고정된 Random Projection Matrix 생성 (한 번만 만들어야 함!)
    # 데이터 차원 계산을 위해 샘플 하나만 먼저 봅니다.

    mean = np.mean(dataset.data, axis=(0, 1), keepdims=True)
    std = np.std(dataset.data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    def transform(x):
        return (x - mean) / std

    features = []
    # indices가 많으면 배치 처리가 좋지만, 여기선 심플하게 loop
    for idx in tqdm(indices, desc="Extracting & Projecting features"):
        sample = dataset[idx]
        
        # 1. Raw Feature (Flatten)
        inputs = transform(sample['inputs'])[:, :, model_config.FORWARD_FEATURES]
        target = transform(sample['target'])[:, :, model_config.TARGET_FEATURES]
        raw_feat = np.concatenate([inputs.reshape(-1), target.reshape(-1)]) # (D_raw,)
        
        # 2. Random Projection (D_raw) @ (D_raw, D_target) -> (D_target)
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
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Extract features
    print("\nExtracting features...")
    train_features = extract_features(train_dataset, range(len(train_dataset)), model_config)
    val_features = extract_features(val_dataset, range(len(val_dataset)), model_config)
    print(f"Train features: {train_features.shape}, Val features: {val_features.shape}")
    
    train_proj, val_sorted = prepare_sliced_metric(
        train_features=train_features,
        val_features=val_features,
        n_projections=100,
        device=device,
        seed=42
    )
    g1 = GumbelSigmoidCoresetOptimizer(train_proj, val_sorted, n_common=4096, device=device)
    # idx1, best1 = g1.optimize(
    #     steps=10000,
    #     lr=5e-2,
    #     tau_start=1.0,
    #     tau_end=0.2,
    #     lambda_sparsity=0.0,
    #     log_every=50
    # )    

    # ===== k sweep: exact-k + uniform-weight SW evaluation =====
    ratios = [0.01] + [i / 10 for i in range(1, 10)] + [1.0]  # 1%,10%,...,90%,100%
    N = train_proj.shape[0]

    # Gumbel 학습 결과로부터 ranking score 생성
    # (optimize()가 best_logits를 저장하므로 접근 가능)
    assert g1.best_logits is not None, "best_logits가 None입니다. optimize()에서 best_logits 업데이트 확인 필요"
    scores = torch.sigmoid(g1.best_logits.detach())  # (N,) on device

    print("\n" + "="*60)
    print("Exact-k sweep (uniform-weight SW)")
    print("="*60)

    results = []
    for r in ratios:
        k = int(round(r * N))
        k = max(1, min(k, N))

        if k == N:
            idx = torch.arange(N, device=device, dtype=torch.long)
        else:
            idx = torch.topk(scores, k=k, largest=True).indices

        d = sw_uniform_exact_k(train_proj, val_sorted, idx, n_common=4096)
        print(f"k={k:6d} ({k/N:6.2%}) | SW={d:.6f}")
        results.append((r, k, d))

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/sw_exactk_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "k", "sw"])
        for r, k, d in results:
            w.writerow([r, k, d])

    print(f"\nSaved sweep results to {csv_path}")

    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    # score_50 = run_fixed_k_experiment(optimizer, 0.5)
    # score_30 = run_fixed_k_experiment(optimizer, 0.3)
    # print(f"Score 50: {score_50:.6f}, Score 30: {score_30:.6f}")
    print(f"\nSaved to results/optimal_coreset_indices.npy")

def run_fixed_k_experiment(optimizer, fixed_ratio=0.5):
    print(f"\n=== Fixed k={fixed_ratio:.0%} Optimization Experiment ===")
    
    # 강제로 k 고정 (k_min = k_max = fixed_ratio)
    original_min = optimizer.k_min
    original_max = optimizer.k_max
    
    optimizer.k_min = fixed_ratio - 0.001
    optimizer.k_max = fixed_ratio + 0.001
    
    # 재초기화 및 최적화
    optimizer._initialize_population_gpu()
    # Feature extraction 다시 할 필요 없음 (이미 되어있음)
    
    # 30 epoch만 짧게 돌려봄
    k, indices = optimizer.optimize(max_iter=30, verbose=True)
    
    print(f"Fixed {fixed_ratio:.0%} Result: SW={optimizer.best_fitness:.6f}")
    
    # 원상복구
    optimizer.k_min = original_min
    optimizer.k_max = original_max
    
    return optimizer.best_fitness


if __name__ == "__main__":
    main()