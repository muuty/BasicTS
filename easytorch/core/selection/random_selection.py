from typing import List
from torch.utils.data import Dataset, DataLoader
from easytorch.core.selection.base import BaseSelection
import random

class RandomSelection(BaseSelection):
    def __init__(self, dataset: Dataset, ratio: float):
        self.dataset = dataset
        self.ratio = ratio

    def select_indices(self) -> List[int]:
        indices = list(range(len(self.dataset)))
        sampled_size = int(len(self.dataset) * self.ratio)
        sampled_indices = random.sample(indices, sampled_size)  # 랜덤 샘플링
        return sampled_indices