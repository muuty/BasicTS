from typing import List
from torch.utils.data import Dataset
from typing import Protocol


class OutlierRemoval(Protocol):
    """Abstract base class for outlier removal methods."""

    def get_normal_indices(self, dataset: Dataset) -> List[int]:
        """Returns a list of indices that are considered inliers (non-outliers)."""
        ...
