"""
Proxy metrics for coreset quality evaluation.

6 metrics in 3 categories:
- Distributional: OT Cost, Sinkhorn Divergence (see coreset/ot_distance.py)
- Combinatorial: FL Objective, Redundancy, Information Gain
- Temporal: Temporal Diversity (H_tod, H_dow)
"""

import numpy as np
import torch
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Combinatorial metrics (similarity-based)
# ---------------------------------------------------------------------------

def compute_fl_objective(
    sim: np.ndarray,
    selected_indices: List[int],
    batch_size: int = 2048,
) -> float:
    """Facility Location objective: Σ_i max_{j∈S} sim(i,j).

    Measures how well each data point is "covered" by its nearest
    representative in the coreset.

    Args:
        sim: (N, N) similarity matrix (e.g. RBF kernel)
        selected_indices: coreset indices
        batch_size: batch for GPU computation

    Returns:
        FL value (higher = better coverage)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = np.array(selected_indices)
    N = sim.shape[0]

    # sim[:, S] is (N, |S|), take max over S for each row
    fl_sum = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sim_batch = torch.from_numpy(sim[start:end][:, S]).to(device).float()
        fl_sum += sim_batch.max(dim=1).values.sum().item()

    return fl_sum


def compute_redundancy(
    sim: np.ndarray,
    selected_indices: List[int],
) -> float:
    """Intra-coreset redundancy: Σ_{i,j∈S} sim(i,j).

    Measures how similar the selected points are to each other.

    Args:
        sim: (N, N) similarity matrix
        selected_indices: coreset indices

    Returns:
        Redundancy value (lower = less redundant)
    """
    S = np.array(selected_indices)
    sim_ss = sim[np.ix_(S, S)]
    return float(np.sum(sim_ss))


def compute_information_gain(
    fl_value: float,
    redundancy_value: float,
    lam: float = 1.0,
) -> float:
    """Information Gain = FL - λ * Redundancy.

    Same structure as GraphCut objective. Measures net information
    after accounting for redundancy.

    Args:
        fl_value: Facility Location objective value
        redundancy_value: Intra-coreset redundancy value
        lam: redundancy penalty weight

    Returns:
        IG value (higher = more net information)
    """
    return fl_value - lam * redundancy_value


def compute_combinatorial_metrics(
    sim: np.ndarray,
    selected_indices: List[int],
    lam: float = 1.0,
) -> Dict[str, float]:
    """Compute all 3 combinatorial metrics at once.

    Args:
        sim: (N, N) similarity matrix
        selected_indices: coreset indices
        lam: redundancy penalty weight for IG

    Returns:
        Dict with keys: fl_objective, redundancy, information_gain
    """
    fl = compute_fl_objective(sim, selected_indices)
    red = compute_redundancy(sim, selected_indices)
    ig = compute_information_gain(fl, red, lam)
    return {
        'fl_objective': fl,
        'redundancy': red,
        'information_gain': ig,
    }


# ---------------------------------------------------------------------------
# Similarity matrix construction (shared with graph_cut.py)
# ---------------------------------------------------------------------------

def build_similarity_rbf(dist_matrix: np.ndarray) -> np.ndarray:
    """RBF kernel with median heuristic: exp(-d^2 / 2σ^2)."""
    N = dist_matrix.shape[0]
    n_sample = min(100000, N * (N - 1) // 2)
    idx_i = np.random.randint(0, N, n_sample)
    idx_j = np.random.randint(0, N, n_sample)
    mask = idx_i != idx_j
    sigma = float(np.median(dist_matrix[idx_i[mask], idx_j[mask]]))
    if sigma == 0:
        sigma = 1.0
    return np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# Temporal diversity metrics
# ---------------------------------------------------------------------------

def compute_temporal_diversity(
    selected_indices: List[int],
    dataset_size: int,
    samples_per_day: int = 288,  # 5-min intervals: 24*60/5 = 288
) -> Dict[str, float]:
    """Temporal diversity via normalized entropy of time-of-day and day-of-week.

    Args:
        selected_indices: coreset indices (0-indexed, time-ordered)
        dataset_size: total number of samples in the dataset
        samples_per_day: samples per day (288 for 5-min intervals)

    Returns:
        Dict with keys: h_tod (24 bins), h_dow (7 bins)
    """
    indices = np.array(selected_indices)

    # Map index to time-of-day bin (0-23) and day-of-week bin (0-6)
    samples_per_hour = samples_per_day // 24  # 12
    tod_bins = (indices % samples_per_day) // samples_per_hour  # 0-23
    dow_bins = (indices // samples_per_day) % 7  # 0-6

    h_tod = _normalized_entropy(tod_bins, 24)
    h_dow = _normalized_entropy(dow_bins, 7)

    return {
        'h_tod': round(h_tod, 4),
        'h_dow': round(h_dow, 4),
    }


def _normalized_entropy(bins: np.ndarray, n_bins: int) -> float:
    """Compute normalized entropy H / log(n_bins). Range [0, 1]."""
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    # Only consider non-zero bins for entropy
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_bins)
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)
