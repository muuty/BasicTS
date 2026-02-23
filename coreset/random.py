from typing import List
from torch.utils.data import Dataset
from coreset.base import BaseSelection
import random

class RandomSelection(BaseSelection):
    def __init__(self, dataset: Dataset, ratio: float, seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed

    def select_indices(self) -> List[int]:
        random.seed(self.seed)
        indices = list(range(len(self.dataset)))
        sampled_size = int(len(self.dataset) * self.ratio)
        sampled_indices = random.sample(indices, sampled_size)
        return sampled_indices