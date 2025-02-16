from scipy.spatial.distance import pdist, squareform
import numpy as np
import hdbscan
from torch.utils.data import Dataset
from selection.base import BaseSelection
from tqdm import tqdm

class HDBSCANSelection(BaseSelection):
    def __init__(self, dataset: Dataset, ratio: float, min_cluster_size: int = 1, min_samples: int = 3, metric: str = "precomputed"):
        """
        HDBSCAN 기반 밀도 중심 샘플링 (precomputed 거리 행렬 사용)
        :param dataset: PyTorch Dataset
        :param ratio: 샘플링할 데이터 비율 (0 ~ 1)
        :param min_samples: HDBSCAN의 min_samples 값
        :param metric: 거리 계산 방식 (기본값: "euclidean", "cosine", "precomputed" 등 가능)
        """
        self.dataset = dataset
        self.ratio = ratio
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric  # 거리 계산 방식

    def select_indices(self) -> list[int]:
        """
        HDBSCAN을 이용하여 밀도가 높은 데이터를 선택하여 인덱스 반환.
        """
        dataset_size = len(self.dataset)
        X = np.array([self.dataset[i]['inputs'][:, :, 0].flatten() for i in range(dataset_size)])
        sample_size = int(len(X) * self.ratio)

        # 1. 거리 행렬을 미리 계산 (사전 계산 방식 적용)
        if self.metric == "precomputed":
            distance_matrix = squareform(pdist(X, metric="euclidean"))  # 거리 행렬 계산
        else:
            distance_matrix = X  # 기존 방식 사용

        # 2. HDBSCAN 실행 (precomputed 사용)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed" if self.metric == "precomputed" else self.metric  # precomputed 적용 여부
        )
        labels = clusterer.fit_predict(distance_matrix)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # 노이즈(-1) 제외

        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

        density_scores = -clusterer.outlier_scores_

        selected_indices = []
        for label, _ in tqdm(sorted_clusters):
            cluster_indices = np.where(labels == label)[0]
            cluster_density = density_scores[cluster_indices]

            sorted_indices = cluster_indices[np.argsort(cluster_density)[:min(len(cluster_indices), sample_size - len(selected_indices))]]
            selected_indices.extend(sorted_indices.tolist())

            if len(selected_indices) >= sample_size:
                break

        return selected_indices[:sample_size]  # 정확히 sample_size 개수만 반환