from typing import List
from torch.utils.data import Dataset
import numpy as np
from selection.outlier_removal.base import OutlierRemoval
from hdbscan import HDBSCAN
class HDBSCANOutlierRemoval(OutlierRemoval):
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 2):
        """
        HDBSCAN 기반 Outlier 제거 클래스.

        Args:
            min_cluster_size (int): 최소 클러스터 크기
            min_samples (int): Outlier 감지를 위한 최소 샘플 개수
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def get_normal_indices(self, dataset: Dataset) -> List[int]:
        """HDBSCAN을 사용하여 Outlier를 제거한 데이터 인덱스 리스트 반환"""

        dataset_size = len(dataset)
        inputs = np.array([self.dataset[i]['inputs'][:,:,0] +
                           self.dataset[i]['target'][:, :, 0]
                           for i in range(dataset_size)])

        # HDBSCAN 클러스터링 수행
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        labels = clusterer.fit_predict(inputs)

        # -1은 Outlier, 나머지는 정상 데이터
        inlier_indices = np.where(labels != -1)[0].tolist()

        return inlier_indices
