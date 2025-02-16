from typing import List
from torch.utils.data import Dataset
from selection.base import BaseSelection
import numpy as np
import kmedoids  # K-Medoids 사용

class KMedoidsSelection(BaseSelection):
    def __init__(self, dataset: Dataset, ratio: float, metric="euclidean", seed: int = 42):
        """
        K-Medoids 기반으로 샘플링하는 클래스.

        Args:
            dataset (Dataset): PyTorch Dataset
            ratio (float): 선택할 데이터 비율
            metric (str): 거리 측정 방식 ("euclidean", "manhattan" 등)
            seed (int): Random seed (기본값: 42)
        """
        self.dataset = dataset
        self.ratio = ratio
        self.metric = metric
        self.seed = seed

    def select_indices(self) -> List[int]:
        """K-Medoids 기반으로 dataset * ratio 개수만큼 샘플링하여 데이터 인덱스를 반환"""

        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        # 데이터 추출 및 Flatten
        inputs = np.array([self.dataset[i]['inputs'][:,:,0].flatten() for i in range(dataset_size)])
        distances = [
            [np.linalg.norm(inputs[i] - inputs[j]) for j in range(dataset_size)]
            for i in range(dataset_size)
        ]



        # K-Medoids 클러스터링 수행
        print(f"Running K-Medoids with {sampled_size} clusters...")
        km = kmedoids.KMedoids(sampled_size, method='fasterpam')
        km.fit(distances)

        return list(km.medoid_indices_)
