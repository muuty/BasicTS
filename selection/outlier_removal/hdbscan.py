from typing import List
from torch.utils.data import Dataset
import numpy as np
from selection.outlier_removal.base import OutlierRemoval
from hdbscan import HDBSCAN
from selection.embedding.base import BaseEmbedding

class HDBSCANOutlierRemoval(OutlierRemoval):
    def __init__(self, min_cluster_size: int, embedding: BaseEmbedding, min_samples: int, removal_ratio: float):
        """
        HDBSCAN 기반 Outlier 제거 클래스.

        Args:
            min_cluster_size (int): 최소 클러스터 크기
            min_samples (int): Outlier 감지를 위한 최소 샘플 개수

        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.removal_ratio = removal_ratio
        self.embedding = embedding

    def get_normal_indices(self, dataset: Dataset) -> List[int]:
        """HDBSCAN을 사용하여 Outlier를 제거한 데이터 인덱스 리스트 반환"""

        dataset_size = len(dataset)
        inputs = np.array([dataset[i]['inputs'][:,:,0] + dataset[i]['target'][:, :, 0]
                           for i in range(dataset_size)])
        inputs = inputs.reshape(dataset_size, -1)

        # Embedding 적용
        if self.embedding is not None:
            inputs = self.embedding.transform(inputs)


        # HDBSCAN 클러스터링 수행
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        clusterer.fit(inputs)

        outlier_scores = clusterer.outlier_scores_
        num_remove = int(len(outlier_scores) * self.removal_ratio)  # 제거할 개수 계산

        # 정확한 개수만 제거하도록 처리
        sorted_indices = np.argsort(outlier_scores)  # Outlier score 기준으로 정렬 (오름차순)
        inlier_indices = sorted_indices[:-num_remove] if num_remove > 0 else sorted_indices

        return inlier_indices.tolist()
