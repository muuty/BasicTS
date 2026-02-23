import numpy as np
import random
from typing import List
import torch
from torch.utils.data import Dataset
from coreset.base import BaseSelection
from coreset.distance import extract_features, compute_distance_matrix


class TotalSimilaritySelection(BaseSelection):
    """
    Facility Location Selection:
    max_{S} sum_{i} max_{j in S} sim(x_i, x_j)

    Greedy algorithm with (1 - 1/e) approximation guarantee.
    """

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

        # Convert distance to similarity: sim = -distance
        similarity_matrix = -distance_matrix
        similarity_matrix = torch.from_numpy(similarity_matrix).float()

        if torch.cuda.is_available():
            similarity_matrix = similarity_matrix.cuda()

        N = similarity_matrix.shape[0]
        selected_indices = []

        # Current max similarity for each point (initialized to -inf)
        current_max_sim = torch.full((N,), float('-inf'), device=similarity_matrix.device)

        for k in range(sampled_size):
            # Marginal gain: gain[j] = sum_i max(0, sim[i,j] - current_max_sim[i])
            improvements = torch.clamp(similarity_matrix - current_max_sim.unsqueeze(1), min=0)  # (N, N)
            marginal_gains = improvements.sum(dim=0)  # (N,)

            # Mask already selected
            for idx in selected_indices:
                marginal_gains[idx] = float('-inf')

            best_idx = marginal_gains.argmax().item()
            selected_indices.append(best_idx)

            current_max_sim = torch.maximum(current_max_sim, similarity_matrix[:, best_idx])

            if (k + 1) % 500 == 0:
                total_coverage = current_max_sim.sum().item()
                print(f"FacilityLocation: {k + 1}/{sampled_size} selected, coverage: {total_coverage:.2f}")

        return selected_indices
