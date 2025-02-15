import numpy as np
import random
from typing import List
from torch.utils.data import Dataset
from easytorch.core.selection.base import BaseSelection
from sklearn.metrics.pairwise import euclidean_distances

class KCenterGreedySelection(BaseSelection):
    """
    Implements k-Center-Greedy sampling to select a diverse subset of the dataset based on input-output distances.
    """

    def __init__(self, dataset: Dataset, ratio: float, seed: int = 42):
        """
        Args:
            dataset (Dataset): PyTorch dataset with `__getitem__` implemented.
            ratio (float): Fraction of data to select.
            seed (int): Random seed for reproducibility.
        """
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed

    def _compute_distance_matrix(self, data_points: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        return euclidean_distances(data_points, data_points)

    def select_indices(self) -> List[int]:
        """Select indices using k-Center-Greedy algorithm."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Step 1: Extract input features from dataset
        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        # Convert dataset into numpy array for distance calculation
        inputs = np.array([self.dataset[i]['inputs'] for i in range(dataset_size)])
        inputs = inputs.reshape(dataset_size, -1)  # Flatten if necessary

        # Step 2: Compute Distance Matrix
        distance_matrix = self._compute_distance_matrix(inputs)

        # Step 3: Initialize with a random point
        first_index = np.random.choice(dataset_size)
        selected_indices = [first_index]
        remaining_indices = set(range(dataset_size)) - set(selected_indices)

        # Step 4: Greedily select k diverse points
        for _ in range(sampled_size - 1):
            # Find the point that is the furthest from the current selected set
            min_distances = np.min(distance_matrix[list(selected_indices), :], axis=0)
            max_dist_index = np.argmax(min_distances[list(remaining_indices)])

            new_index = list(remaining_indices)[max_dist_index]
            selected_indices.append(new_index)
            remaining_indices.remove(new_index)

        return selected_indices