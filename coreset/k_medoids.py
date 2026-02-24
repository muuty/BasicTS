import numpy as np
import random
from typing import List
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from easytorch.device import to_device




def compute_gain_matrix(D, medoids, nearest_dist, nearest_medoid, non_medoids):
    N, k = D.shape[0], medoids.shape[0]
    H = non_medoids.shape[0]

    D_h = D[:, non_medoids]  # (N, H)
    gain_matrix = torch.zeros((H, k), device=D.device)
    old_total = nearest_dist.sum()

    for j in range(k):
        mask = torch.arange(k, device=D.device) != j
        alt_medoids = medoids[mask]
        D_alt = D[:, alt_medoids]  # (N, k-1)
        alt_dist = D_alt.min(dim=1)[0]  # (N,)

        candidate_new_dists = torch.min(
            D_h.T.unsqueeze(2),  # (H, N, 1)
            alt_dist.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        ).squeeze(2)  # (H, N)

        gain = old_total - candidate_new_dists.sum(dim=1)  # (H,)
        gain_matrix[:, j] = gain

    return gain_matrix

def fasterpam_gpu_vectorized(D: torch.Tensor, k: int, max_iter: int = 5) -> torch.Tensor:
    N = D.shape[0]
    device = D.device

    medoids = torch.randperm(N, device=device)[:k]
    D_medoids = D[:, medoids]
    nearest_dist, nearest_medoid = D_medoids.min(dim=1)

    for it in range(max_iter):
        non_medoids = torch.tensor([i for i in range(N) if i not in medoids], device=device)
        gain_matrix = compute_gain_matrix(D, medoids, nearest_dist, nearest_medoid, non_medoids)  # (H, k)

        best_h_idx, best_j = torch.nonzero(gain_matrix == gain_matrix.max(), as_tuple=True)
        best_h = non_medoids[best_h_idx[0]]

        if gain_matrix[best_h_idx[0], best_j[0]] <= 0:
            break

        medoids[best_j[0]] = best_h
        D_medoids = D[:, medoids]
        nearest_dist, nearest_medoid = D_medoids.min(dim=1)

    return medoids

def get_distance_matrix(x_np: np.ndarray, device: str = 'cuda', batch_size: int = 128) -> np.ndarray:
    N, D = x_np.shape
    dist_mat = torch.empty((N, N), dtype=torch.float32, device='cpu')  # CPU에 저장

    # Precompute norms of all samples (on CPU for now)
    x_norm_cpu = np.sum(x_np**2, axis=1)

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        x_batch = torch.from_numpy(x_np[i:end_i]).to(device).float()         # (B, D)
        x_batch_norm = torch.from_numpy(x_norm_cpu[i:end_i]).to(device).float().unsqueeze(1)  # (B, 1)

        # Compute against full x_np (CPU-side → GPU-side in chunks too)
        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            x_ref = torch.from_numpy(x_np[j:end_j]).to(device).float()  # (B, D)
            x_ref_norm = torch.from_numpy(x_norm_cpu[j:end_j]).to(device).float().unsqueeze(0)  # (1, B)

            dot = x_batch @ x_ref.T  # (B1, B2)
            dist_sq = x_batch_norm + x_ref_norm - 2 * dot
            dist_sq = torch.clamp(dist_sq, min=0.0)
            dist = torch.sqrt(dist_sq).cpu()  # move to CPU

            dist_mat[i:end_i, j:end_j] = dist

    return dist_mat.numpy()        # GPU → CPU



def kmedoids_selection(data: np.ndarray, k: int, device: str = 'cuda', seed: int = 42):
    """K-Medoids selection using FasterPAM."""
    random.seed(seed)
    np.random.seed(seed)
    if k == 1:
        # return every index
        return list(range(len(data)))

    
    D = get_distance_matrix(data, device=device)
    D_tensor = torch.from_numpy(D).to(device).float()
    medoid_indices = fasterpam_gpu_vectorized(D_tensor, k=k, max_iter=20)
    medoid_indices = medoid_indices.cpu().numpy()
    
    selection = np.zeros(len(data), dtype=bool)
    selection[medoid_indices] = True
    
    return medoid_indices.tolist()


# class KMedoidsSelection(BaseSelection):
#     def __init__(self, dataset: Dataset, ratio: float, embedding_model: BaseEmbedding | None, model_config: dict, seed: int = 42):
#         self.dataset = dataset
#         self.ratio = ratio
#         self.embedding_model = embedding_model
#         self.model_config = model_config
#         self.seed = seed

#     def select_indices(self) -> List[int]:
#         random.seed(self.seed)
#         np.random.seed(self.seed)
#         dataset_size = len(self.dataset)
#         sampled_size = int(dataset_size * self.ratio)
        
#         mean = np.mean(self.dataset.data, axis=(0,1), keepdims=True)
#         std = np.std(self.dataset.data, axis=(0,1), keepdims=True)
#         std[std == 0] = 1.0
        
#         def transform(input_data):
#             return (input_data - mean) / std
        
#         # Convert dataset into numpy array (FIXED: concatenate instead of +)
#         features = []
#         for i in range(dataset_size):
#             sample = self.dataset[i]
#             inputs = transform(sample['inputs'])[:, :, self.model_config.FORWARD_FEATURES]
#             target = transform(sample['target'])[:, :, self.model_config.TARGET_FEATURES]
            
#             # Flatten and concatenate
#             feat = np.concatenate([inputs.reshape(-1), target.reshape(-1)])
#             features.append(feat)
        
#         features = np.array(features, dtype=np.float32)  # (dataset_size, feat_dim)
        
#         # Step 2: Apply Embedding
#         if self.embedding_model is not None:
#             features = self.embedding_model.transform(features)
        
#         # Step 3: Compute pairwise distance matrix on GPU
#         distance_matrix = get_distance_matrix(features)
        
#         # Step 4: Run FasterPAM algorithm
#         distance_matrix = torch.from_numpy(distance_matrix).to('cuda').float()
#         medoid_indices = fasterpam_gpu_vectorized(distance_matrix, k=sampled_size, max_iter=5)
        
#         return medoid_indices.cpu().tolist()