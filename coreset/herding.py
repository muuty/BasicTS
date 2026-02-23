import numpy as np
import random
from typing import List
from torch.utils.data import Dataset
from coreset.base import BaseSelection
from coreset.distance import extract_features, get_features_by_type


class HerdingSelection(BaseSelection):
    """
    Herding (Mean Matching): greedy selection to match the global mean.

    For each step, select the sample that minimizes the L2 distance
    between the running mean of selected samples and the global mean.

    Reference: Welling (2009) "Herding Dynamic Weights for Online Learning"
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

        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        inputs, targets = extract_features(self.dataset, self.model_config)
        features = get_features_by_type(inputs, targets, self.distance_type)
        # (N, D) where D depends on distance_type

        global_mean = features.mean(axis=0)  # (D,)

        selected_indices = []
        selected_sum = np.zeros_like(global_mean)
        remaining = set(range(dataset_size))

        for k in range(sampled_size):
            best_idx = None
            best_score = -np.inf

            target_sum = (k + 1) * global_mean

            for idx in remaining:
                candidate_sum = selected_sum + features[idx]
                score = -np.linalg.norm(candidate_sum - target_sum)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_indices.append(best_idx)
            selected_sum += features[best_idx]
            remaining.remove(best_idx)

            if (k + 1) % 1000 == 0:
                print(f"Herding: {k + 1}/{sampled_size} selected")

        return selected_indices
