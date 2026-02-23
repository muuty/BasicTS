"""
k-Center Greedy selection for coreset.

Minimizes the maximum distance from any point to its nearest selected center.
Greedy 2-approximation algorithm.
"""

import numpy as np
import random
import torch
from typing import List
from torch.utils.data import Dataset
from coreset.base import BaseSelection
from coreset.distance import extract_features, compute_distance_matrix


class KCenterGreedySelection(BaseSelection):

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
        dist_np = compute_distance_matrix(inputs, targets, self.distance_type)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dist = torch.from_numpy(dist_np).to(device).float()

        # Initialize with random point
        first_idx = torch.randint(0, dataset_size, (1,), device=device).item()
        selected = [first_idx]

        # Track min distance from each point to nearest center (incremental update)
        min_dist = dist[first_idx].clone()

        for k in range(1, sampled_size):
            # Point with largest min-distance to any selected center
            next_idx = min_dist.argmax().item()
            selected.append(next_idx)
            # Update: min_dist[i] = min(min_dist[i], dist[i, next_idx])
            min_dist = torch.minimum(min_dist, dist[next_idx])

            if (k + 1) % 1000 == 0:
                print(f"k-Center: {k+1}/{sampled_size}, max radius: {min_dist.max().item():.4f}")

        return selected
