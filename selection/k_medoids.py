from typing import List
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset
from selection.base import BaseSelection
from kmedoids import KMedoids
from selection.embedding.base import BaseEmbedding
from sklearn.metrics.pairwise import euclidean_distances


class KMedoidsSelection(BaseSelection):
    def __init__(self, dataset: Dataset, embedding_model: BaseEmbedding, ratio: float, metric: str = "euclidean"):
        """
        K-Medoids-based Selection with Precomputed Distance Matrix

        :param dataset: PyTorch Dataset
        :param ratio: The fraction of the dataset to be selected.
        :param metric: Distance metric for precomputed calculations (e.g., "euclidean", "cosine").
        """
        self.dataset = dataset
        self.ratio = ratio
        self.metric = metric  # Determines how distances are calculated
        self.embedding_model = embedding_model

    def select_indices(self) -> List[int]:
        """
        Perform K-Medoids clustering and return the indices of the selected medoids.

        :return: List of selected data indices.
        """
        dataset_size = len(self.dataset)
        sample_size = int(dataset_size * self.ratio)

        # Extract data from dataset
        inputs = np.array([self.dataset[i]['inputs'][:,:,0] +
                           self.dataset[i]['target'][:, :, 0]
                           for i in range(dataset_size)])
        inputs = inputs.reshape(dataset_size, -1)  # Flatten if necessary
        # Step 2: Apply Embedding
        embedded_data = self.embedding_model.transform(inputs)  # ✅ t-SNE 등 사용 가능

        distance_matrix = euclidean_distances(embedded_data, embedded_data)


        # Run K-Medoids with precomputed distance matrix
        kmedoids = KMedoids(n_clusters=sample_size, metric="precomputed", init="random", random_state=42)
        kmedoids.fit(distance_matrix)

        # Return selected medoid indices
        return kmedoids.medoid_indices_.tolist()