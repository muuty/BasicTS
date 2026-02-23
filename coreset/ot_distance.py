"""
Optimal Transport distance/divergence for coreset quality evaluation.

Measures distributional distance between coreset and full dataset.
- sinkhorn_distance: vanilla regularized OT (biased)
- sinkhorn_divergence: debiased version (Feydy et al., NeurIPS 2019)
  SD(P, Q) = OT_ε(P, Q) - 0.5 * OT_ε(P, P) - 0.5 * OT_ε(Q, Q)
"""

import numpy as np
import torch
from typing import Optional
from sklearn.decomposition import PCA


def compute_cost_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost matrix. C[i,j] = ||X[i] - Y[j]||^2."""
    X_sqnorm = (X ** 2).sum(dim=1, keepdim=True)
    Y_sqnorm = (Y ** 2).sum(dim=1, keepdim=True)
    C = X_sqnorm + Y_sqnorm.T - 2 * X @ Y.T
    return torch.clamp(C, min=0)


def _sinkhorn_cost(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
    threshold: float = 1e-6
) -> float:
    """Compute regularized OT cost <P*, C> via Sinkhorn iterations."""
    n, m = X.shape[0], Y.shape[0]
    device = X.device

    a = torch.ones(n, device=device) / n
    b = torch.ones(m, device=device) / m

    C = compute_cost_matrix(X, Y)
    K = torch.exp(-C / epsilon)

    u = torch.ones(n, device=device)
    v = torch.ones(m, device=device)

    for _ in range(max_iter):
        u_prev = u.clone()
        u = a / (K @ v + 1e-10)
        v = b / (K.T @ u + 1e-10)
        if torch.max(torch.abs(u - u_prev)) < threshold:
            break

    P = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(P * C).item()


def sinkhorn_divergence(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
    threshold: float = 1e-6
) -> float:
    """
    Debiased Sinkhorn divergence.

    SD(P, Q) = OT_ε(P, Q) - 0.5 * OT_ε(P, P) - 0.5 * OT_ε(Q, Q)

    Properties:
    - SD(P, P) = 0 (proper divergence)
    - SD(P, Q) >= 0
    - ε → 0: converges to Wasserstein distance
    - ε → ∞: converges to MMD

    Reference: Feydy et al., "Interpolating between Optimal Transport
    and MMD using Sinkhorn Divergences", NeurIPS 2019.
    """
    ot_pq = _sinkhorn_cost(X, Y, epsilon, max_iter, threshold)
    ot_pp = _sinkhorn_cost(X, X, epsilon, max_iter, threshold)
    ot_qq = _sinkhorn_cost(Y, Y, epsilon, max_iter, threshold)
    return max(0.0, ot_pq - 0.5 * ot_pp - 0.5 * ot_qq)


# Legacy wrapper for backward compatibility
def sinkhorn_distance(X, Y, a=None, b=None, epsilon=0.1, max_iter=100, threshold=1e-6):
    """Vanilla Sinkhorn distance (biased). Prefer sinkhorn_divergence instead."""
    cost = _sinkhorn_cost(X, Y, epsilon, max_iter, threshold)
    return cost, None


class OTDistanceCalculator:
    """Coreset quality evaluator using Optimal Transport."""

    def __init__(self, pca_dim: int = 50, epsilon: float = 0.1,
                 max_iter: int = 100, device: str = 'cuda'):
        self.pca_dim = pca_dim
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.pca = None

    def fit_pca(self, features: np.ndarray) -> None:
        actual_dim = min(self.pca_dim, features.shape[1], features.shape[0])
        self.pca = PCA(n_components=actual_dim)
        self.pca.fit(features)
        print(f"PCA fitted: {features.shape[1]}D -> {actual_dim}D "
              f"(explained variance: {self.pca.explained_variance_ratio_.sum():.4f})")

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        return self.pca.transform(features)

    def _prepare(self, coreset_features: np.ndarray, full_features: np.ndarray,
                 subsample_full: Optional[int] = 5000):
        """PCA transform + subsample + move to device."""
        if self.pca is None:
            self.fit_pca(full_features)

        coreset_pca = self.transform(coreset_features)
        full_pca = self.transform(full_features)

        if subsample_full and len(full_pca) > subsample_full:
            indices = np.random.choice(len(full_pca), subsample_full, replace=False)
            full_pca = full_pca[indices]

        X = torch.from_numpy(coreset_pca).float().to(self.device)
        Y = torch.from_numpy(full_pca).float().to(self.device)
        return X, Y

    def compute_distance(self, coreset_features: np.ndarray,
                         full_features: np.ndarray,
                         subsample_full: Optional[int] = 5000) -> float:
        """Vanilla Sinkhorn distance (biased)."""
        X, Y = self._prepare(coreset_features, full_features, subsample_full)
        return _sinkhorn_cost(X, Y, self.epsilon, self.max_iter)

    def compute_divergence(self, coreset_features: np.ndarray,
                           full_features: np.ndarray,
                           subsample_full: Optional[int] = 5000) -> float:
        """Debiased Sinkhorn divergence (preferred)."""
        X, Y = self._prepare(coreset_features, full_features, subsample_full)
        return sinkhorn_divergence(X, Y, self.epsilon, self.max_iter)


def compute_ot_from_indices(
    dataset,
    selected_indices: list,
    model_config: dict,
    distance_type: str = 'euclidean',
    pca_dim: int = 50,
    epsilon: float = 0.1,
    subsample_full: int = 5000,
    debiased: bool = True
) -> float:
    """
    Convenience function: compute OT metric from selected indices.

    Uses shared feature extraction from coreset.distance module,
    supporting temporal/spatial/combined features.

    Args:
        dataset: TimeSeriesForecastingDataset
        selected_indices: coreset indices
        model_config: model config (FORWARD_FEATURES, TARGET_FEATURES)
        distance_type: feature type for OT computation
        pca_dim: PCA dimension reduction
        epsilon: Sinkhorn regularization
        subsample_full: subsample full dataset for efficiency
        debiased: if True, use Sinkhorn divergence; else vanilla distance

    Returns:
        OT metric value (lower = better coreset)
    """
    from coreset.distance import extract_features, get_features_by_type

    inputs, targets = extract_features(dataset, model_config)
    features = get_features_by_type(inputs, targets, distance_type)

    coreset_features = features[selected_indices]
    full_features = features

    calculator = OTDistanceCalculator(pca_dim=pca_dim, epsilon=epsilon)

    if debiased:
        return calculator.compute_divergence(coreset_features, full_features, subsample_full)
    else:
        return calculator.compute_distance(coreset_features, full_features, subsample_full)


# Backward compatibility alias
compute_ot_distance_from_indices = compute_ot_from_indices
