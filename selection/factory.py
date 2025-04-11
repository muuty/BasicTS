from selection.base import BaseSelection
from selection.coverage_centric import CoverageCentricSelection
from selection.embedding.base import BaseEmbedding
from selection.graph_cut import GraphCutSMISelection
from selection.random_selection import RandomSelection
from selection.recent_selection import RecentSelection
from selection.k_center_greedy import KCenterGreedySelection
from selection.k_medoids import KMedoidsSelection
from torch.utils.data import Dataset
import numpy as np

def get_selection(type: str,
                  selection_ratio: float,
                  embedding_model: BaseEmbedding | None,
                  dataset: Dataset,
                  data_config: dict,
                  model_config: dict
                  ) -> BaseSelection | None:

    if not type or selection_ratio == 1.0:
        return None

    if type == 'random':
        return RandomSelection(dataset=dataset, ratio=selection_ratio)
    elif type == 'recent':
        return RecentSelection(dataset=dataset, ratio=selection_ratio)
    elif type == 'k_center_greedy':
        return KCenterGreedySelection(dataset=dataset,
                                      ratio=selection_ratio,
                                      model_config=model_config,
                                      embedding_model=embedding_model)
    elif type == 'k_medoids':
        return KMedoidsSelection(dataset=dataset,
                                 ratio=selection_ratio,
                                 embedding_model=embedding_model)
    elif type == 'coverage_centric':
        scores = np.loadtxt(data_config['IMPORTANCE_SCORES_PATH'], delimiter=",", usecols=1)
        return CoverageCentricSelection(dataset=dataset,
                                        ratio=selection_ratio,
                                        importance_scores=scores)
    elif type == 'graph_cut':
        return GraphCutSMISelection(dataset=dataset,
                                        ratio=selection_ratio)
    else:
        raise ValueError('Unknown selection type!')
