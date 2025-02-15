from umap import UMAP
import numpy as np
from selection.embedding.base import BaseEmbedding

class UMAPEmbedding(BaseEmbedding):
    """
    UMAP embedding module that projects high-dimensional data into a lower-dimensional space.
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    def transform(self, data: np.ndarray) -> np.ndarray:
        umap = UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist, random_state=self.random_state)
        return umap.fit_transform(data)