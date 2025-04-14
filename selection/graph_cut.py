from typing import List, Optional
import numpy as np
import torch

from torch.utils.data import Dataset
from selection.utils import get_distance_matrix, get_normalized_input,distance_to_similarity_rbf

class GraphCutSMISelection:
    """
    Graph Cut Submodular Mutual Information (SMI) based Coreset Selection.
    Maximizes mutual information between selected subset and its complement
    using greedy forward selection.
    """

    def __init__(
            self,
            dataset: Dataset,
            ratio: float,
            model_config: dict,
            seed: int = 42
    ):
        self.dataset = dataset  # shape: (n, d)
        self.ratio = ratio
        self.model_config = model_config
        self.seed = seed

    def select_indices(self) -> List[int]:
        np.random.seed(self.seed)
        N = len(self.dataset)
        k = int(N * self.ratio)

        # Step 1: Compute distance matrix
        inputs = get_normalized_input(self.dataset, self.model_config)
        distance_matrix = get_distance_matrix(inputs)
        distance_matrix= torch.from_numpy(distance_matrix).to('cuda').float()
        similarity_matrix = distance_to_similarity_rbf(distance_matrix)

        selected_mask = torch.zeros(N, dtype=torch.bool, device=similarity_matrix.device)
        gain = torch.zeros(N, device=similarity_matrix.device)

        selected = []

        for _ in range(k):
            remaining_mask = ~selected_mask

            if selected:
                # similarity to selected set (N,)
                gain = similarity_matrix[:, selected_mask].sum(dim=1)
            else:
                gain.zero_()

            # similarity to remaining set
            gain += similarity_matrix[:, remaining_mask].sum(dim=1)

            # exclude already selected
            gain[selected_mask] = -float('inf')

            best_idx = torch.argmax(gain).item()
            selected.append(best_idx)
            selected_mask[best_idx] = True

        return selected
