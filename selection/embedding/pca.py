## Write PCA Embedding
import numpy as np
from sklearn.decomposition import PCA
from selection.embedding.base import BaseEmbedding

class PCAEmbedding(BaseEmbedding):
    """PCA embedding class."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, data: np.ndarray):
        embedding = PCA(n_components=self.n_components).fit_transform(data)
        return embedding
