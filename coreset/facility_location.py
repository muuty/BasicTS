"""
Facility Location Selection: maximize total coverage via nearest representative.

Objective: f(S) = sum_i max_{j in S} sim(i,j)

Each data point's "coverage" is determined by its most similar selected representative.
Submodular → greedy gives (1 - 1/e) approximation guarantee.

Greedy marginal gain:
  delta(e|S) = sum_i max(sim(i,e) - current_max[i], 0)
  i.e., only points where e is closer than all previously selected benefit.

O(k*N) total with running max tracker.

Equivalent to k-medoids objective (minimize sum of distances to nearest medoid)
but solved via greedy with theoretical guarantees instead of PAM local search.

Reference: DeepCore (Guo et al., 2022), Wei et al. (2015)
"""

import numpy as np
import random
from typing import List
import torch
from torch.utils.data import Dataset
from coreset.base import BaseSelection
from coreset.distance import (
    extract_features, compute_distance_matrix, parse_distance_type,
    _get_repr_features, get_temporal_features, get_spatial_features,
)
from coreset.graph_cut import _build_similarity_rbf, _build_similarity_cosine


class FacilityLocationSelection(BaseSelection):

    def __init__(self, dataset: Dataset, ratio: float, model_config: dict,
                 distance_type: str = 'euclidean', seed: int = 42):
        self.dataset = dataset
        self.ratio = ratio
        self.model_config = model_config
        self.distance_type = distance_type
        self.seed = seed

    def select_indices(self) -> List[int]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        dataset_size = len(self.dataset)
        sampled_size = int(dataset_size * self.ratio)

        inputs, targets = extract_features(self.dataset, self.model_config)

        repr_name, metric = parse_distance_type(self.distance_type)

        if metric == 'cosine':
            if repr_name == 'combined':
                sim_t = _build_similarity_cosine(
                    get_temporal_features(inputs, targets))
                sim_s = _build_similarity_cosine(
                    get_spatial_features(inputs, targets))
                similarity_np = 0.5 * sim_t + 0.5 * sim_s
            else:
                features = _get_repr_features(inputs, targets, repr_name)
                similarity_np = _build_similarity_cosine(features)
        else:
            dist_np = compute_distance_matrix(inputs, targets, self.distance_type)
            similarity_np = _build_similarity_rbf(dist_np, self.seed)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sim = torch.from_numpy(similarity_np).to(device).float()

        N = sim.shape[0]
        selected_indices = []
        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # Track max similarity from each point to its nearest selected representative
        # f(S) = sum_i current_max[i]
        current_max = torch.zeros(N, device=device)

        for k in range(sampled_size):
            # Marginal gain of adding e:
            #   delta(e|S) = sum_i max(sim(i,e) - current_max[i], 0)
            # Vectorized: for each candidate e, compute gain over all i
            gains = torch.clamp(sim - current_max.unsqueeze(1), min=0).sum(dim=0)
            gains[selected_mask] = float('-inf')

            best_idx = gains.argmax().item()
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True

            # Update: current_max[i] = max(current_max[i], sim(i, best_idx))
            current_max = torch.max(current_max, sim[:, best_idx])

            if (k + 1) % 500 == 0:
                obj_val = current_max.sum().item()
                print(f"FacilityLocation: {k+1}/{sampled_size}, "
                      f"objective={obj_val:.2f}")

        return selected_indices
