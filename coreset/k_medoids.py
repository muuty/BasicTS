import numpy as np
import random
from typing import List
import torch
from torch.utils.data import Dataset
from coreset.base import BaseSelection
from coreset.distance import extract_features, compute_distance_matrix


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
    def __init__(self, dataset: Dataset, ratio: float, model_config: dict,
                 distance_type: str = 'euclidean', seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.model_config = model_config
        self.distance_type = distance_type
        self.seed = seed

    def select_indices(self) -> List[int]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        inputs, targets = extract_features(self.dataset, self.model_config)
        distance_matrix = compute_distance_matrix(inputs, targets, self.distance_type)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        distance_matrix = torch.from_numpy(distance_matrix).to(device).float()
        medoid_indices = fasterpam_gpu_vectorized(distance_matrix, k=sampled_size, max_iter=5)

        return medoid_indices.cpu().tolist()
