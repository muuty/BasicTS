from typing import List
import numpy as np
from typing import Protocol

class BaseEmbedding(Protocol):
    """
    Abstract base class for embedding models.
    """

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply embedding to the given data.

        Args:
            data (np.ndarray): Input data of shape (N, D)

        Returns:
            np.ndarray: Transformed embedding of shape (N, d)
        """
        ...