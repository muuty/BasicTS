from easytorch.core.selection.base import BaseSelection
from easytorch.core.selection.random_selection import RandomSelection
from easytorch.core.selection.k_random_greedy import KCenterGreedySelection
from torch.utils.data import Dataset, DataLoader



def get_selection(type: str, selection_ratio: float, dataset: Dataset) -> BaseSelection:
    """Get selection instance.

    Args:
        type (str): Selection type.
        dataset: Dataset.

    Returns:
        BaseSelection: Selection instance.
    """
    if not type or selection_ratio == 1.0:
        return None

    if type == 'random':
        return RandomSelection(dataset, selection_ratio)
    if type == 'k_center_greedy':
        return KCenterGreedySelection(dataset, selection_ratio)
    else:
        raise ValueError('Unknown selection type!')