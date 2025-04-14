from typing import List
import numpy as np
import torch

from torch.utils.data import Dataset
from selection.utils import get_distance_matrix, get_normalized_input, distance_to_similarity_rbf

class FacilityLocationSelection:
    """
    Facility Location-based Coreset Selection.
    Maximizes total similarity of each point to its nearest selected point.
    Greedy forward selection.
    """

    def __init__(
            self,
            dataset: Dataset,
            ratio: float,
            model_config: dict,
            seed: int = 42
    ):
        self.dataset = dataset
        self.ratio = ratio
        self.model_config = model_config
        self.seed = seed

    def select_indices(self) -> List[int]:
        np.random.seed(self.seed)
        N = len(self.dataset)
        k = int(N * self.ratio)

        # Step 1: Compute similarity matrix
        inputs = get_normalized_input(self.dataset, self.model_config)
        distance_matrix = get_distance_matrix(inputs)
        distance_matrix = torch.from_numpy(distance_matrix).to('cuda').float()
        similarity_matrix = distance_to_similarity_rbf(distance_matrix)

        # Step 2: Initialize
        selected_mask = torch.zeros(N, dtype=torch.bool, device=similarity_matrix.device)
        coverage = torch.zeros(N, device=similarity_matrix.device)
        selected = []

        for _ in range(k):
            # For all remaining candidates, compute marginal gain
            marginal_gain = similarity_matrix - coverage.unsqueeze(1)  # (N, N)
            marginal_gain = torch.clamp(marginal_gain, min=0.0)
            gain = marginal_gain.sum(dim=0)  # (N,)

            # Exclude already selected
            gain[selected_mask] = -float('inf')

            best_idx = torch.argmax(gain).item()
            selected.append(best_idx)
            selected_mask[best_idx] = True

            # Update coverage
            coverage = torch.max(coverage, similarity_matrix[:, best_idx])

        return selected