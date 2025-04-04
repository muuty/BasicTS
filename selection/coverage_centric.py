import random
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional


class CoverageCentricSelection:
    """
    Implements Coverage-Centric Coreset Selection (CCS) for regression.
    Uses stratified sampling over error-based importance scores.
    """

    def __init__(self, dataset: Dataset, ratio: float, importance_scores: np.ndarray,
                 seed: int = 42, beta: float = 0.05, num_strata: int = 10):
        self.dataset = dataset
        self.ratio = ratio
        self.importance_scores = importance_scores
        self.seed = seed
        self.beta = beta
        self.num_strata = num_strata

    def select_indices(self) -> List[int]:
        """
        Select indices using CCS: prune hardest samples, stratify remaining,
        and sample proportionally across strata to maintain coverage.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        n = len(self.dataset)
        keep_n = int(n * self.ratio)
        cutoff_n = int(n * self.beta)

        # Step 1: Sort scores descending, prune top beta hardest
        sorted_indices = np.argsort(-self.importance_scores)
        remaining_indices = sorted_indices[cutoff_n:]

        # Step 2: Stratify remaining scores
        remaining_scores = self.importance_scores[remaining_indices]
        bins = np.linspace(remaining_scores.min(), remaining_scores.max(), self.num_strata + 1)
        strata = [[] for _ in range(self.num_strata)]

        for idx in remaining_indices:
            score = self.importance_scores[idx]
            bin_idx = np.searchsorted(bins, score, side='right') - 1
            bin_idx = min(bin_idx, self.num_strata - 1)
            strata[bin_idx].append(idx)

        # Step 3: Sample from strata, prioritizing underrepresented bins
        selected_indices = []
        budget = keep_n
        active_strata = [s for s in strata if len(s) > 0]

        while budget > 0 and active_strata:
            # Pick the smallest stratum (coverage-centric)
            active_strata.sort(key=len)
            stratum = active_strata.pop(0)
            take = min(len(stratum), max(1, budget // len(active_strata) if active_strata else budget))
            sampled = np.random.choice(stratum, size=take, replace=False).tolist()
            selected_indices.extend(sampled)
            budget -= len(sampled)

        return selected_indices