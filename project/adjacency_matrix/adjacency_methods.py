"""
Adjacency Matrix Generation Methods for GNN Traffic Prediction.

Registry pattern: each method is registered via @register decorator.
All methods return a raw numpy adjacency matrix (N, N) before model-specific normalization.
"""

import numpy as np
import pandas as pd

from basicts.utils import load_pkl, load_dataset_desc


ADJ_METHOD_REGISTRY = {}


def register(name):
    """Register an adjacency method by name."""
    def decorator(fn):
        ADJ_METHOD_REGISTRY[name] = fn
        return fn
    return decorator


def get_adjacency(method, dataset_name, **kwargs):
    """Main entry point. Returns raw adjacency matrix (np.ndarray, shape [N, N]).

    Args:
        method: Name of the registered adjacency method.
        dataset_name: Name of the dataset (e.g., 'SAN_BERNARDINO', 'CONTRA_COSTA').
        **kwargs: Method-specific parameters.

    Returns:
        np.ndarray: Raw adjacency matrix of shape (N, N), dtype float32.
    """
    return ADJ_METHOD_REGISTRY[method](dataset_name, **kwargs)


def _load_sensor_coords(dataset_name):
    """Load sensor Lat/Lng from metadata.csv.

    Returns:
        lats, lngs: np.ndarray of shape (N,) each.
    """
    meta = pd.read_csv(f"datasets/{dataset_name}/metadata.csv")
    return meta['Lat'].values, meta['Lng'].values


def _haversine_distance_matrix(lats, lngs):
    """Compute pairwise haversine distances in km.

    Args:
        lats, lngs: np.ndarray of shape (N,), in degrees.

    Returns:
        np.ndarray of shape (N, N), pairwise distances in km.
    """
    lat1 = np.radians(lats[:, None])
    lat2 = np.radians(lats[None, :])
    dlat = lat2 - lat1
    dlng = np.radians(lngs[None, :]) - np.radians(lngs[:, None])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def _gaussian_kernel(dist, sigma):
    """Apply Gaussian kernel: exp(-d^2 / (2*sigma^2))."""
    return np.exp(-dist ** 2 / (2 * sigma ** 2))


def _build_distance_adj(dataset_name, sigma=2.0, threshold=0.1):
    """Build Gaussian kernel adjacency from sensor coordinates.

    Args:
        dataset_name: Dataset name.
        sigma: Gaussian kernel bandwidth in km.
        threshold: Minimum weight to keep an edge.

    Returns:
        np.ndarray: Adjacency matrix of shape (N, N), dtype float32.
    """
    lats, lngs = _load_sensor_coords(dataset_name)
    dist = _haversine_distance_matrix(lats, lngs)
    adj = _gaussian_kernel(dist, sigma)
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


# ============================================================
# Distance-based methods (Gaussian kernel from coordinates)
# sigma=2km fixed, threshold varies for sparsity control
# ============================================================

SIGMA_KM = 2.0  # Gaussian kernel bandwidth

@register("distance")
def distance_adjacency(dataset_name, **kwargs):
    """Standard distance-based Gaussian kernel adjacency.

    Constructed from sensor Lat/Lng coordinates via haversine distance.
    A_ij = exp(-d(i,j)^2 / (2*sigma^2)) if A_ij >= 0.1, else 0.
    sigma=2km, threshold=0.1 → avg_deg ~50-60.

    Reference: DCRNN (Li et al., 2018), GWNet (Wu et al., 2019).
    """
    return _build_distance_adj(dataset_name, sigma=SIGMA_KM, threshold=0.1)


@register("distance_sparse")
def distance_sparse_adjacency(dataset_name, **kwargs):
    """Sparse distance-based adjacency (high threshold).

    Same Gaussian kernel as 'distance' but with higher threshold,
    keeping only strong (nearby) connections.
    sigma=2km, threshold=0.5 → avg_deg ~25-30.
    """
    return _build_distance_adj(dataset_name, sigma=SIGMA_KM, threshold=0.5)


@register("distance_dense")
def distance_dense_adjacency(dataset_name, **kwargs):
    """Dense distance-based adjacency (low threshold).

    Same Gaussian kernel as 'distance' but with lower threshold,
    including weaker (farther) connections.
    sigma=2km, threshold=0.01 → avg_deg ~85-95.
    """
    return _build_distance_adj(dataset_name, sigma=SIGMA_KM, threshold=0.01)


@register("identity")
def identity_adjacency(dataset_name, **kwargs):
    """Identity matrix (no spatial information baseline).

    Each node only connects to itself. Tests whether spatial graph
    provides any benefit over pure temporal modeling.
    """
    desc = load_dataset_desc(dataset_name)
    n = desc['num_nodes']
    return np.eye(n, dtype=np.float32)
