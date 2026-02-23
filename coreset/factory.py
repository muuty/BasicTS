from coreset.selection import BaseSelection
from coreset.random import RandomSelection
from coreset.k_medoids import KMedoidsSelection
from coreset.k_center import KCenterGreedySelection
from coreset.herding import HerdingSelection
from coreset.recent import RecentSelection
from coreset.stride import StrideSelection
from coreset.total_similarity import TotalSimilaritySelection
from coreset.graph_cut import GraphCutSelection
from coreset.facility_location import FacilityLocationSelection
from torch.utils.data import Dataset


def get_selection(type: str,
                  selection_ratio: float,
                  dataset: Dataset,
                  model_config: dict,
                  distance_type: str = 'euclidean',
                  similarity_type: str = 'rbf',
                  seed: int = 42
                  ) -> BaseSelection | None:

    if not type or selection_ratio == 1.0:
        return DefaultSelection(dataset)

    if type == 'random':
        return RandomSelection(dataset, selection_ratio, seed=seed)

    if type == 'k_medoids':
        return KMedoidsSelection(dataset,
                                 ratio=selection_ratio,
                                 model_config=model_config,
                                 distance_type=distance_type,
                                 seed=seed)

    if type == 'k_center':
        return KCenterGreedySelection(dataset,
                                      ratio=selection_ratio,
                                      model_config=model_config,
                                      distance_type=distance_type,
                                      seed=seed)

    if type == 'herding':
        return HerdingSelection(dataset,
                                ratio=selection_ratio,
                                model_config=model_config,
                                distance_type=distance_type,
                                seed=seed)

    if type == 'recent':
        return RecentSelection(dataset,
                               ratio=selection_ratio,
                               seed=seed)

    if type == 'stride':
        return StrideSelection(dataset,
                               ratio=selection_ratio,
                               seed=seed)

    if type == 'total_similarity':
        return TotalSimilaritySelection(dataset,
                                        ratio=selection_ratio,
                                        model_config=model_config,
                                        distance_type=distance_type,
                                        seed=seed)

    if type == 'graph_cut':
        return GraphCutSelection(dataset,
                                 ratio=selection_ratio,
                                 model_config=model_config,
                                 distance_type=distance_type,
                                 similarity_type=similarity_type,
                                 seed=seed)

    if type == 'facility_location':
        return FacilityLocationSelection(dataset,
                                         ratio=selection_ratio,
                                         model_config=model_config,
                                         distance_type=distance_type,
                                         seed=seed)

    raise ValueError(f'Unknown selection type: {type}')


class DefaultSelection(BaseSelection):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def select_indices(self) -> list[int]:
        return list(range(len(self.dataset)))
