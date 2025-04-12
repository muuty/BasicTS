from typing import List
import numpy as np
from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset

class StrideSelection:
    """
    Time-aware coreset selection method using fixed stride.
    Selects k samples spaced as evenly as possible across the dataset.
    """

    def __init__(
        self,
        dataset: TimeSeriesForecastingDataset,
        ratio: float,
        seed: int = 42
    ):
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed

    def select_indices(self) -> List[int]:
        np.random.seed(self.seed)
        N = len(self.dataset)
        k = int(N * self.ratio)

        # Use linspace to evenly distribute k points from range(N)
        indices = np.linspace(0, N - 1, k, dtype=int).tolist()
        return indices
