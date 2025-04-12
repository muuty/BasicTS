import numpy as np
import random
from typing import List
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from selection.base import BaseSelection
from selection.embedding.base import BaseEmbedding
from easytorch.device import to_device


def pairwise_distance_matrix(x_np: np.ndarray, device: str = 'cuda', batch_size: int = 1024) -> np.ndarray:
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

    return dist_mat.numpy()


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


class KMedoidsSelection(BaseSelection):
    """
    Implements k-Medoids selection using precomputed distance matrix and FasterPAM algorithm.
    """

    def __init__(self, 
            dataset: Dataset, ratio: float, 
            embedding_model: BaseEmbedding, 
            model_config: dict,
            seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.embedding_model = embedding_model
        self.seed = seed
        self.model_config = model_config

    def select_indices(self) -> List[int]:
        random.seed(self.seed)
        np.random.seed(self.seed)

        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        mean = np.mean(self.dataset.data, axis=(0,1), keepdims=True)
        std = np.std(self.dataset.data, axis=(0,1), keepdims=True)
        std[std == 0] = 1.0

        def transform(input_data):
            return (input_data - mean) / std

        # Convert dataset into numpy array
        inputs = np.array([
            transform(self.dataset[i]['inputs'])[:,:,self.model_config.MODEL.FORWARD_FEATURES] +
            transform(self.dataset[i]['target'])[:,:,self.model_config.MODEL.TARGET_FEATURES]
            for i in range(dataset_size)])
        inputs = inputs.reshape(dataset_size, -1)

        # Step 2: Apply Embedding
        if self.embedding_model is not None:
            inputs = self.embedding_model.transform(inputs)

        # Step 3: Compute pairwise distance matrix on GPU
        D = pairwise_distance_matrix(inputs)  # (N, N)

        # Step 4: Run FasterPAM algorithm
        D = torch.from_numpy(D).to('cuda').float()
        medoid_indices = fasterpam_gpu_vectorized(D, k=sampled_size, max_iter=5)

        return medoid_indices.cpu().tolist()