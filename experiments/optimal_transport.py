import csv
import ot
import sys
import numpy as np
import os
from easydict import EasyDict
import torch
import tempfile
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basicts.data import TimeSeriesForecastingDataset
from coreset.factory import get_selection
from coreset.base import measure_time


def get_dataset(dataset_name: str, input_len: int, output_len: int, train_val_test_ratio: tuple, mode: str):
    return TimeSeriesForecastingDataset(dataset_name=dataset_name, input_len=input_len, output_len=output_len, train_val_test_ratio=train_val_test_ratio, mode=mode)


def extract_feature(sample, transform, model_config):
    inputs = transform(sample['inputs'])[:, :, model_config.FORWARD_FEATURES]
    target = transform(sample['target'])[:, :, model_config.TARGET_FEATURES]
    
    feat = np.concatenate([
        inputs.reshape(-1),
        target.reshape(-1)
    ])
    return feat.astype(np.float32)


@measure_time
def get_coreset_method(type: str, selection_ratio: float, dataset: Dataset, model_config: dict):
    selection_method = get_selection(type=type, selection_ratio=selection_ratio, embedding_model=None, dataset=dataset, model_config=model_config)
    return selection_method.select_indices()

import torch
import numpy as np
from tqdm import tqdm
import ot
import os
import tempfile

def extract_features_batched(
    dataset,
    indices: list,
    model_config: dict,
    batch_size: int = 2000,
    use_memmap: bool = True,
    memmap_path: str = None
):
    """Feature 추출 (memmap)"""
    mean = np.mean(dataset.data, axis=(0, 1), keepdims=True)
    std = np.std(dataset.data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    
    def transform(x):
        return (x - mean) / std
    
    sample = dataset[indices[0]]
    inputs = transform(sample['inputs'])[:, :, model_config.FORWARD_FEATURES]
    target = transform(sample['target'])[:, :, model_config.TARGET_FEATURES]
    feat_dim = inputs.reshape(-1).shape[0] + target.reshape(-1).shape[0]
    
    n = len(indices)
    print(f"Feature dim: {feat_dim}, n_samples: {n}")
    
    if use_memmap:
        if memmap_path is None:
            mmap_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
            memmap_path = mmap_file.name
            mmap_file.close()
        features = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n, feat_dim))
    else:
        features = np.empty((n, feat_dim), dtype=np.float32)
    
    for i in tqdm(range(0, n, batch_size), desc="Extracting features"):
        end_i = min(i + batch_size, n)
        for j, idx in enumerate(indices[i:end_i]):
            sample = dataset[idx]
            inputs = transform(sample['inputs'])[:, :, model_config.FORWARD_FEATURES]
            target = transform(sample['target'])[:, :, model_config.TARGET_FEATURES]
            feat = np.concatenate([inputs.reshape(-1), target.reshape(-1)])
            features[i + j] = feat
        
        if use_memmap and i % (batch_size * 10) == 0:
            features.flush()
    
    if use_memmap:
        features.flush()
    
    return features, memmap_path, feat_dim


def sliced_wasserstein_gpu(
    train_features,
    val_features,
    n_train: int,
    n_val: int,
    feat_dim: int,
    n_projections: int = 200,
    batch_size: int = 2000,
    device: str = 'cuda'
) -> float:
    """
    GPU 기반 Sliced Wasserstein
    - Cost matrix 생성 안함
    - Projection → 1D sort → Wasserstein
    """
    print(f"Computing Sliced Wasserstein with {n_projections} projections...")
    
    # Random projections (GPU)
    torch.manual_seed(42)
    projections = torch.randn(n_projections, feat_dim, device=device, dtype=torch.float32)
    projections = projections / projections.norm(dim=1, keepdim=True)
    
    # Project train data (batch 단위)
    print("Projecting train data...")
    train_projected = torch.zeros(n_projections, n_train, device=device)
    
    for i in tqdm(range(0, n_train, batch_size)):
        end_i = min(i + batch_size, n_train)
        batch = torch.from_numpy(np.array(train_features[i:end_i])).to(device).float()
        proj = projections @ batch.T  # (n_proj, batch_size)
        train_projected[:, i:end_i] = proj
        del batch, proj
    
    # Project val data (batch 단위)
    print("Projecting val data...")
    val_projected = torch.zeros(n_projections, n_val, device=device)
    
    for i in tqdm(range(0, n_val, batch_size)):
        end_i = min(i + batch_size, n_val)
        batch = torch.from_numpy(np.array(val_features[i:end_i])).to(device).float()
        proj = projections @ batch.T
        val_projected[:, i:end_i] = proj
        del batch, proj
    
    # Sort
    print("Sorting projected values...")
    train_sorted, _ = torch.sort(train_projected, dim=1)
    val_sorted, _ = torch.sort(val_projected, dim=1)
    
    # Interpolate to same size (for unequal n_train, n_val)
    print("Computing 1D Wasserstein...")
    if n_train != n_val:
        n_common = max(n_train, n_val)
        
        # Linear interpolation indices
        train_idx = torch.linspace(0, n_train - 1, n_common, device=device)
        val_idx = torch.linspace(0, n_val - 1, n_common, device=device)
        
        # Interpolate
        train_idx_floor = train_idx.long().clamp(0, n_train - 2)
        train_idx_ceil = (train_idx_floor + 1).clamp(0, n_train - 1)
        train_frac = (train_idx - train_idx_floor.float()).unsqueeze(0)
        
        train_interp = (
            train_sorted[:, train_idx_floor] * (1 - train_frac) +
            train_sorted[:, train_idx_ceil] * train_frac
        )
        
        val_idx_floor = val_idx.long().clamp(0, n_val - 2)
        val_idx_ceil = (val_idx_floor + 1).clamp(0, n_val - 1)
        val_frac = (val_idx - val_idx_floor.float()).unsqueeze(0)
        
        val_interp = (
            val_sorted[:, val_idx_floor] * (1 - val_frac) +
            val_sorted[:, val_idx_ceil] * val_frac
        )
        
        train_sorted = train_interp
        val_sorted = val_interp
    
    # 1D Wasserstein = mean absolute difference
    w1_per_proj = torch.mean(torch.abs(train_sorted - val_sorted), dim=1)
    sw_distance = torch.mean(w1_per_proj).item()
    
    return sw_distance


def compute_sliced_wasserstein(
    train_dataset,
    train_indices: list,
    val_dataset,
    val_indices: list,
    model_config: dict,
    n_projections: int = 200,
    feature_batch_size: int = 2000,
    projection_batch_size: int = 2000,
    device: str = 'cuda',
    cleanup: bool = True
) -> float:
    """전체 파이프라인"""
    
    # Feature 추출
    print("=" * 40)
    print("Step 1: Extracting train features...")
    train_features, train_mmap, feat_dim = extract_features_batched(
        train_dataset, train_indices, model_config,
        feature_batch_size, use_memmap=True
    )
    
    print("Step 2: Extracting val features...")
    val_features, val_mmap, _ = extract_features_batched(
        val_dataset, val_indices, model_config,
        feature_batch_size, use_memmap=True
    )
    
    # Sliced Wasserstein
    print("=" * 40)
    print("Step 3: Computing Sliced Wasserstein...")
    sw_distance = sliced_wasserstein_gpu(
        train_features, val_features,
        n_train=len(train_indices),
        n_val=len(val_indices),
        feat_dim=feat_dim,
        n_projections=n_projections,
        batch_size=projection_batch_size,
        device=device
    )
    
    # Cleanup
    del train_features, val_features
    
    if cleanup:
        for path in [train_mmap, val_mmap]:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up: {path}")
    
    return sw_distance


if __name__ == "__main__":
    import csv
    import sys
    from easydict import EasyDict
    
    sys.path.append(os.path.abspath(__file__ + '/../..'))
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from basicts.data import TimeSeriesForecastingDataset
    from coreset.factory import get_selection
    
    def get_dataset(dataset_name, input_len, output_len, train_val_test_ratio, mode):
        return TimeSeriesForecastingDataset(
            dataset_name=dataset_name, input_len=input_len, output_len=output_len,
            train_val_test_ratio=train_val_test_ratio, mode=mode
        )
    
    dataset_name = 'xtraffic/SAN_BERNARDINO'
    train_dataset = get_dataset(dataset_name, 12, 12, (0.6, 0.2, 0.2), 'train')
    val_dataset = get_dataset(dataset_name, 12, 12, (0.6, 0.2, 0.2), 'valid')
    
    model_config = EasyDict()
    model_config.FORWARD_FEATURES = [0, 1, 2]
    model_config.TARGET_FEATURES = [0]
    
    val_indices = list(range(len(val_dataset)))
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    results = []
    # for ratio in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]:
    for ratio in [1.0]:
        print(f"\n{'='*60}")
        print(f"Selection ratio: {ratio}")
        print(f"{'='*60}")
        
        selection_method = get_selection(
            type='random',
            selection_ratio=ratio,
            embedding_model=None,
            dataset=train_dataset,
            model_config=model_config
        )
        if selection_method is None:
            coreset_indices = list(range(len(train_dataset)))
        else:
            coreset_indices = selection_method.select_indices()
        print(f"Coreset size: {len(coreset_indices)}")
        
        sw_distance = compute_sliced_wasserstein(
            train_dataset=train_dataset,
            train_indices=coreset_indices,
            val_dataset=val_dataset,
            val_indices=val_indices,
            model_config=model_config,
            n_projections=200,
            feature_batch_size=2000,
            projection_batch_size=2000,
            device='cuda',
            cleanup=True
        )
        
        print(f"Sliced Wasserstein distance: {sw_distance:.6f}")
        results.append({
            'ratio': ratio,
            'sw_distance': sw_distance,
            'coreset_size': len(coreset_indices)
        })
    
    # 결과 저장
    os.makedirs('results', exist_ok=True)
    with open('results/sliced_wasserstein_gpu.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'ratio', 'coreset_size', 'sw_distance'])
        for r in results:
            writer.writerow([dataset_name, r['ratio'], r['coreset_size'], r['sw_distance']])
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"Ratio {r['ratio']:.1f}: SW = {r['sw_distance']:.6f}")