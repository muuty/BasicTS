"""
GraphCut Selection: maximize cross-partition similarity with redundancy penalty.

Objective: f(S) = sum_i sum_{j in S} sim(i,j) - lambda * sum_{i,j in S} sim(i,j)

When lambda=1, this simplifies to the graph cut value:
  f(S) = sum_{i in V\\S, j in S} sim(i,j)
i.e., total edge weight crossing the partition (S, V\\S).

Greedy algorithm with marginal gain:
  delta(e|S) = column_sum[e] - lambda * (2 * sum_{i in S} sim(i,e) + sim(e,e))
  Since sim(e,e) is constant across candidates, it's dropped from argmax.

O(k*N) total: column sums are precomputed, only sim_to_selected updates per step.

Supports multiple similarity kernels:
  - rbf:    exp(-d^2 / 2*sigma^2), sigma via median heuristic. Range [0, 1].
  - cosine: 0.5 + 0.5 * cos(v1, v2), DeepCore default. Range [0, 1].

Reference: Iyer & Bilmes (2021), DeepCore (Guo et al., 2022)
"""

import numpy as np
import random
from typing import List
import torch
from torch.utils.data import Dataset
from coreset.base import BaseSelection
from coreset.distance import (
    extract_features, compute_distance_matrix, parse_distance_type,
    _get_repr_features, get_temporal_features, get_spatial_features,
)


def _build_similarity_rbf(dist_np: np.ndarray, seed: int = 42) -> np.ndarray:
    """RBF kernel: exp(-d^2 / 2*sigma^2) with median heuristic for sigma."""
    N = dist_np.shape[0]
    np.random.seed(seed)
    n_sample = min(100000, N * (N - 1) // 2)
    idx_i = np.random.randint(0, N, n_sample)
    idx_j = np.random.randint(0, N, n_sample)
    mask = idx_i != idx_j
    sigma = float(np.median(dist_np[idx_i[mask], idx_j[mask]]))
    if sigma == 0:
        sigma = 1.0
    print(f"GraphCut: RBF sigma = {sigma:.4f}")
    return np.exp(-dist_np ** 2 / (2 * sigma ** 2))


def _build_similarity_cosine(features: np.ndarray) -> np.ndarray:
    """Cosine similarity normalized to [0, 1]: 0.5 + 0.5 * cos(v1, v2).
    Same formula as DeepCore's cossim_np."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = features / norms
    cos_sim = normalized @ normalized.T  # (N, N), values in [-1, 1]
    return (0.5 + 0.5 * cos_sim).astype(np.float32)


class GraphCutSelection(BaseSelection):

    def __init__(self, dataset: Dataset, ratio: float, model_config: dict,
                 distance_type: str = 'euclidean', similarity_type: str = 'rbf',
                 lam: float = 1.0, seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.model_config = model_config
        self.distance_type = distance_type
        self.similarity_type = similarity_type
        self.lam = lam
        self.seed = seed

    def select_indices(self) -> List[int]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        inputs, targets = extract_features(self.dataset, self.model_config)

        repr_name, metric = parse_distance_type(self.distance_type)

        if metric == 'cosine':
            # Cosine pipeline: compute cosine similarity directly on features
            if repr_name == 'combined':
                # Avoid dimension imbalance: compute separately and average
                sim_t = _build_similarity_cosine(
                    get_temporal_features(inputs, targets))
                sim_s = _build_similarity_cosine(
                    get_spatial_features(inputs, targets))
                similarity_np = 0.5 * sim_t + 0.5 * sim_s
            else:
                features = _get_repr_features(inputs, targets, repr_name)
                similarity_np = _build_similarity_cosine(features)
        else:
            # L2 pipeline: distance matrix → RBF kernel
            dist_np = compute_distance_matrix(inputs, targets, self.distance_type)
            similarity_np = _build_similarity_rbf(dist_np, self.seed)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sim = torch.from_numpy(similarity_np).to(device).float()

        N = sim.shape[0]
        selected_indices = []
        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # Precompute column sums: representation term is modular (constant per element)
        column_sums = sim.sum(dim=0)  # (N,)

        # Redundancy: track cumulative similarity to selected set
        sim_to_selected = torch.zeros(N, device=device)

        for k in range(sampled_size):
            # Standard GraphCut marginal gain:
            #   delta(e|S) = column_sum[e] - lambda * 2 * sum_{i in S} sim(i,e)
            # (sim(e,e) dropped — constant, doesn't affect argmax)
            marginal_gains = column_sums - self.lam * 2 * sim_to_selected
            marginal_gains[selected_mask] = float('-inf')

            best_idx = marginal_gains.argmax().item()
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True

            # Update redundancy tracker
            sim_to_selected += sim[:, best_idx]

            if (k + 1) % 500 == 0:
                cut_val = column_sums[selected_mask].sum().item()
                redundancy = sim_to_selected[selected_mask].sum().item()
                print(f"GraphCut: {k+1}/{sampled_size}, "
                      f"rep={cut_val:.2f}, red={redundancy:.2f}")

        return selected_indices
