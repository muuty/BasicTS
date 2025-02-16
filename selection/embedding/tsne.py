from sklearn.manifold import TSNE
from selection.embedding.base import BaseEmbedding
import numpy as np
class TSNEEmbedding(BaseEmbedding):
    def __init__(self, n_components=2, perplexity=100, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.max_iter = 500

    def transform(self, data: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, random_state=self.random_state, max_iter=self.max_iter)
        return tsne.fit_transform(data)
