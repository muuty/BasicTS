import torch
import numpy as np

def generate_projections(dim: int, n_projections: int, device: str = 'cuda'):
    """
    1. Projection Matrix 생성 (Random Unit Vectors)
    - 한 번만 생성하여 재사용하기 위함
    """
    # (D, n_proj)
    projections = torch.randn(dim, n_projections, device=device)
    # 정규화 (반지름이 1인 구 위로 투영)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    return projections

def project_data(data: torch.Tensor, projections: torch.Tensor):
    """
    2. 데이터 투영 (Matrix Multiplication)
    Returns: (N, n_proj)
    """
    return data @ projections

def compute_batch_sw_distance(
    source_proj_batch: torch.Tensor, 
    target_proj_sorted: torch.Tensor, 
    p: int = 2,
    device: str = 'cuda'
):
    """
    3. 배치 단위 Sliced Wasserstein Distance 계산
    - Projection은 이미 수행된 상태로 입력받음 (재사용성 확보)
    
    Args:
        source_proj_batch: (n_rabbits, k, n_proj) - 투영은 됐지만 정렬은 안 된 Source
        target_proj_sorted: (M, n_proj) - 투영되고 정렬까지 된 Target (Validation)
        p: 1 for L1, 2 for L2
    """
    # ---------------------------------------------------------
    # 4. 정렬 (Sorting) -> (n_rabbits, k, n_proj)
    # ---------------------------------------------------------
    # dim=1 (k 축) 기준으로 정렬
    source_sorted, _ = torch.sort(source_proj_batch, dim=1) 
    
    # ---------------------------------------------------------
    # 5. 샘플 개수 보간 (Interpolation / Quantile Matching)
    # ---------------------------------------------------------
    n_rabbits, k, n_proj = source_sorted.shape
    M = target_proj_sorted.shape[0]
    
    if k != M:
        # Target 길이(M)에 맞춰 Source(k)를 늘리거나 줄임
        # (Batch 차원 보존하며 인덱싱)
        indices = torch.linspace(0, k - 1, M, device=device).long()
        source_interpolated = source_sorted[:, indices, :] # (n_rabbits, M, n_proj)
    else:
        source_interpolated = source_sorted

    # ---------------------------------------------------------
    # 6. 거리 계산 (평균)
    # ---------------------------------------------------------
    # Target은 (M, n_proj) -> (1, M, n_proj)로 브로드캐스팅하여 연산
    diff = source_interpolated - target_proj_sorted.unsqueeze(0)
    
    if p == 1:
        # Mean over M (dim=1) and Projections (dim=2)
        dist = torch.mean(torch.abs(diff), dim=(1, 2))
    elif p == 2:
        dist = torch.mean(torch.pow(diff, 2), dim=(1, 2))
        dist = torch.sqrt(dist)
    else:
        dist = torch.mean(torch.pow(torch.abs(diff), p), dim=(1, 2))
        dist = torch.pow(dist, 1/p)
        
    return dist # (n_rabbits,) 형태의 텐서 반환