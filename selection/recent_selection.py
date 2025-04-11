from typing import List
from torch.utils.data import Dataset
from selection.base import BaseSelection

class RecentSelection(BaseSelection):
    def __init__(self, dataset: Dataset, ratio: float):
        """
        가장 최근 데이터를 기반으로 샘플링하는 클래스.

        Args:
            dataset (Dataset): PyTorch Dataset
            ratio (float): 선택할 데이터 비율 (예: 0.1은 10% 선택)
        """
        self.dataset = dataset
        self.ratio = ratio

    def select_indices(self) -> List[int]:
        """
        가장 최근의 데이터 중 ratio 비율만큼 선택한 인덱스를 반환한다.
        Dataset이 시간 순으로 정렬되어 있다고 가정함.

        Returns:
            List[int]: 선택된 데이터 인덱스 리스트
        """
        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        # 가장 마지막 sampled_size 개의 인덱스 선택
        return list(range(dataset_size - sampled_size, dataset_size))
