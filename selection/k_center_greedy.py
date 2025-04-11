import numpy as np
import random
from typing import List
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from selection.base import BaseSelection
from selection.embedding.base import BaseEmbedding
from sklearn.metrics.pairwise import euclidean_distances

class KCenterGreedySelection(BaseSelection):
    """
    Implements k-Center-Greedy sampling using a given embedding space.
    """

    def __init__(self, 
            dataset: Dataset, ratio: float, 
            embedding_model: BaseEmbedding, 
            model_config: dict,
            seed: int = 42):
        """
        Args:
            dataset (Dataset): PyTorch dataset with `__getitem__` implemented.
            ratio (float): Fraction of data to select.
            embedding_model (BaseEmbedding): Embedding module to transform the input data.
            seed (int): Random seed for reproducibility.
        """
        self.dataset = dataset
        self.ratio = ratio
        self.embedding_model = embedding_model
        self.seed = seed
        self.model_config = model_config

    def _compute_distance_matrix(self, data_points: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        return euclidean_distances(data_points, data_points)

    def select_indices(self) -> List[int]:
        """Select indices using k-Center-Greedy in embedding space."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Step 1: Extract input features from dataset
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
        inputs = inputs.reshape(dataset_size, -1)  # Flatten if necessary

        # Step 2: Apply Embedding
        if self.embedding_model is not None:
            inputs = self.embedding_model.transform(inputs)  # ✅ t-SNE 등 사용 가능

        # Step 3: Compute Distance Matrix in Embedded Space
        distance_matrix = self._compute_distance_matrix(inputs)

        # Step 4: Initialize with a random point
        first_index = np.random.choice(dataset_size)
        selected_indices = [first_index]
        remaining_indices = set(range(dataset_size)) - set(selected_indices)

        # Step 5: Greedily select k diverse points in embedded space
        for _ in range(sampled_size - 1):
            # Find the point that is the furthest from the current selected set
            min_distances = np.min(distance_matrix[list(selected_indices), :], axis=0)
            max_dist_index = np.argmax(min_distances[list(remaining_indices)])

            new_index = list(remaining_indices)[max_dist_index]
            selected_indices.append(new_index)
            remaining_indices.remove(new_index)

        return selected_indices
