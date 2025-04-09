from typing import List, Optional
import numpy as np

class GraphCutSMISelection:
    """
    Graph Cut Submodular Mutual Information (SMI) based Coreset Selection.
    Maximizes mutual information between selected subset and its complement
    using greedy forward selection.
    """

    def __init__(self, dataset: np.ndarray, ratio: float, seed: int = 42,
                 precomputed_similarity: Optional[np.ndarray] = None):
        self.dataset = dataset  # shape: (n, d)
        self.ratio = ratio
        self.seed = seed
        self.similarity_matrix = precomputed_similarity  # shape: (n, n) or None

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Computes pairwise Euclidean similarity between time series samples.
        For each sample, flatten `inputs[:,:,0]` to 1D vector.
        """
        dataset_size = len(self.dataset)

        # Step 1: Extract input vectors
        inputs = np.array([
            self.dataset[i]['inputs'][:, :, 0].flatten()
            for i in range(dataset_size)
        ])  # shape: (N, D)

        # Step 2: Compute pairwise Euclidean distances (slower, but explicit)
        distances = np.array([
            [np.linalg.norm(inputs[i] - inputs[j]) for j in range(dataset_size)]
            for i in range(dataset_size)
        ])  # shape: (N, N)

        # Step 3: Convert distances to similarity (negative distance)
        similarity = -distances  # Higher similarity = closer

        return similarity

    def select_indices(self) -> List[int]:
        np.random.seed(self.seed)
        n = len(self.dataset)
        k = int(n * self.ratio)

        # Compute similarity matrix if not provided
        if self.similarity_matrix is None:
            self.similarity_matrix = self.compute_similarity_matrix()

        selected = set()
        remaining = set(range(n))

        for _ in range(k):
            max_gain = -np.inf
            best_idx = -1

            for i in remaining:
                Ai = selected | {i}
                Bi = remaining - {i}
                if not Bi:
                    continue
                gain = np.sum(self.similarity_matrix[np.ix_(list(Ai), list(Bi))]) * 2
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i

            selected.add(best_idx)
            remaining.remove(best_idx)

        return list(selected)
