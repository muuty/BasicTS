from sklearn.manifold import TSNE
from selection.embedding.base import BaseEmbedding
import numpy as np


class RawEmbedding(BaseEmbedding):
    def __init__(self):
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data
