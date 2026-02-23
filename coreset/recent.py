from typing import List
from torch.utils.data import Dataset
from coreset.base import BaseSelection


class RecentSelection(BaseSelection):
    """
    Recent Selection: 시간적으로 가장 최근 데이터를 선택.

    Traffic forecasting에서 최근 패턴이 미래 예측에 더 유용할 수 있다는 가정.
    인덱스가 클수록 최근 데이터라고 가정.
    """

    def __init__(self, dataset: Dataset, ratio: float, seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed  # Not used, but kept for interface consistency

    def select_indices(self) -> List[int]:
        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        # 가장 최근 데이터 선택 (마지막 n개)
        start_idx = dataset_size - sampled_size
        selected_indices = list(range(start_idx, dataset_size))

        return selected_indices
