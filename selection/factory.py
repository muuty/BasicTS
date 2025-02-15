from selection.base import BaseSelection
from selection.embedding.base import BaseEmbedding
from selection.random_selection import RandomSelection
from selection.k_random_greedy import KCenterGreedySelection
from torch.utils.data import Dataset


def get_selection(type: str,
                  selection_ratio: float,
                  embedding_model: BaseEmbedding | None,
                  dataset: Dataset) -> BaseSelection | None:

    if not type or selection_ratio == 1.0:
        return None

    if type == 'random':
        return RandomSelection(dataset, selection_ratio)
    if type == 'k_center_greedy':
        return KCenterGreedySelection(dataset, selection_ratio, embedding_model=embedding_model)
    else:
        raise ValueError('Unknown selection type!')