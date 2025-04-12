import numpy as np
from typing import List
from torch.utils.data import Dataset
from selection.base import BaseSelection
from selection.embedding.base import BaseEmbedding


class HerdingSelection(BaseSelection):
    """
    Implements the Herding algorithm to select representative samples
    based on embedding features (no labels required).
    """

    def __init__(
        self,
        dataset: Dataset,
        ratio: float,
        embedding_model: BaseEmbedding,
        model_config: dict,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.ratio = ratio
        self.embedding_model = embedding_model
        self.model_config = model_config
        self.seed = seed

    def select_indices(self) -> List[int]:
        np.random.seed(self.seed)

        dataset_size = len(self.dataset)
        selected_size = int(dataset_size * self.ratio)

        # Step 1: Normalize input data
        mean = np.mean(self.dataset.data, axis=(0, 1), keepdims=True)
        std = np.std(self.dataset.data, axis=(0, 1), keepdims=True)
        std[std == 0] = 1.0

        def transform(input_data):
            return (input_data - mean) / std

        # Step 2: Extract flattened input features
        inputs = np.array([
            transform(self.dataset[i]['inputs'])[:, :, self.model_config.MODEL.FORWARD_FEATURES] +
            transform(self.dataset[i]['target'])[:, :, self.model_config.MODEL.TARGET_FEATURES]
            for i in range(dataset_size)
        ])
        inputs = inputs.reshape(dataset_size, -1)

        # Step 3: Apply embedding
        if self.embedding_model is not None:
            inputs = self.embedding_model.transform(inputs)  # shape: (N, D)

        # Step 4: Herding Algorithm
        mu = np.mean(inputs, axis=0)
        w = np.zeros_like(mu)
        selected_indices = []
        already_selected = set()

        for _ in range(selected_size):
            # Score = inner product with current weight
            scores = inputs @ w
            scores[list(already_selected)] = -np.inf
            index = np.argmax(scores)

            selected_indices.append(index)
            already_selected.add(index)

            # Herding update
            w = w + (mu - inputs[index])

        return [int(index) for index in selected_indices]
