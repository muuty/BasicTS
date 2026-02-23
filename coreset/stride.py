import numpy as np
from typing import List
from torch.utils.data import Dataset
from coreset.base import BaseSelection


class StrideSelection(BaseSelection):
    """
    Stride Selection: select samples at regular intervals.
    Ensures uniform temporal coverage across the training period.
    """

    def __init__(self, dataset: Dataset, ratio: float, seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed  # not used, kept for interface consistency

    def select_indices(self) -> List[int]:
        N = len(self.dataset)
        k = int(N * self.ratio)
        # Evenly spaced indices from 0 to N-1
        indices = np.linspace(0, N - 1, k, dtype=int).tolist()
        return indices
