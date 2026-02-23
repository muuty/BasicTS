"""
Distance functions for coreset selection in spatio-temporal traffic data.

Decomposed into representation x metric:
  representation: raw, temporal, spatial, combined
  metric:         l2 (euclidean), cosine

Supports 8 distance types:
  L2 pipeline (euclidean + RBF for graph_cut):
  - euclidean: L2 on flattened [T*N*F] features (raw representation)
  - temporal:  L2 on mean-over-nodes [T, F] features
  - spatial:   L2 on mean-over-time [N, F] features
  - combined:  normalized L2_temporal + L2_spatial average

  Cosine pipeline (cosine distance, cosine similarity for graph_cut):
  - cosine_raw:      cosine distance on flattened features
  - cosine_temporal:  cosine distance on temporal features
  - cosine_spatial:   cosine distance on spatial features
  - cosine_combined:  normalized cosine_temporal + cosine_spatial average
"""

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def extract_features(dataset: Dataset, model_config) -> tuple:
    """
    Extract normalized input/target features from dataset.

    Returns:
        inputs_all:  (N, T_in, Nodes, len(FORWARD_FEATURES))
        targets_all: (N, T_out, Nodes, len(TARGET_FEATURES))
    """
    dataset_size = len(dataset)

    mean = np.mean(dataset.data, axis=(0, 1), keepdims=True)
    std = np.std(dataset.data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    inputs_list = []
    targets_list = []
    for i in range(dataset_size):
        sample = dataset[i]
        inp = ((sample['inputs'] - mean) / std)[:, :, model_config.FORWARD_FEATURES]
        tgt = ((sample['target'] - mean) / std)[:, :, model_config.TARGET_FEATURES]
        inputs_list.append(inp)
        targets_list.append(tgt)

    inputs_all = np.array(inputs_list, dtype=np.float32)
    targets_all = np.array(targets_list, dtype=np.float32)
    return inputs_all, targets_all


# ---------------------------------------------------------------------------
# Feature projection functions
# ---------------------------------------------------------------------------

def get_flat_features(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Flatten and concatenate inputs/targets. Shape: (N, T_in*Nodes*F + T_out*Nodes*F)."""
    N = inputs.shape[0]
    return np.concatenate([inputs.reshape(N, -1), targets.reshape(N, -1)], axis=1)


def get_temporal_features(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean over nodes, then flatten. Shape: (N, T_in*F + T_out*F).
    Captures temporal variation patterns independent of specific sensor locations."""
    N = inputs.shape[0]
    inp_t = inputs.mean(axis=2)   # (N, T_in, F)
    tgt_t = targets.mean(axis=2)  # (N, T_out, F)
    return np.concatenate([inp_t.reshape(N, -1), tgt_t.reshape(N, -1)], axis=1)


def get_spatial_features(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean over time, then flatten. Shape: (N, Nodes*F + Nodes*F).
    Captures spatial distribution patterns independent of specific time steps."""
    N = inputs.shape[0]
    inp_s = inputs.mean(axis=1)   # (N, Nodes, F)
    tgt_s = targets.mean(axis=1)  # (N, Nodes, F)
    return np.concatenate([inp_s.reshape(N, -1), tgt_s.reshape(N, -1)], axis=1)


def get_features_by_type(inputs: np.ndarray, targets: np.ndarray,
                         distance_type: str) -> np.ndarray:
    """Get features appropriate for the given distance type.

    For methods that operate in feature space (e.g., herding moment matching),
    this maps distance_type to the corresponding feature representation.
    For 'combined', concatenates normalized temporal + spatial features.
    """
    if distance_type == 'euclidean':
        return get_flat_features(inputs, targets)
    elif distance_type == 'temporal':
        return get_temporal_features(inputs, targets)
    elif distance_type == 'spatial':
        return get_spatial_features(inputs, targets)
    elif distance_type == 'combined':
        temporal = get_temporal_features(inputs, targets)
        spatial = get_spatial_features(inputs, targets)
        # Normalize each to unit variance before concatenating
        t_std = temporal.std(axis=0, keepdims=True)
        s_std = spatial.std(axis=0, keepdims=True)
        t_std[t_std == 0] = 1.0
        s_std[s_std == 0] = 1.0
        return np.concatenate([temporal / t_std, spatial / s_std], axis=1)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


# ---------------------------------------------------------------------------
# Representation / metric parsing
# ---------------------------------------------------------------------------

def parse_distance_type(distance_type: str):
    """Parse distance_type into (representation, metric).

    Examples:
        'euclidean'        -> ('raw', 'l2')
        'temporal'         -> ('temporal', 'l2')
        'combined'         -> ('combined', 'l2')
        'cosine_raw'       -> ('raw', 'cosine')
        'cosine_temporal'  -> ('temporal', 'cosine')
        'cosine_combined'  -> ('combined', 'cosine')
    """
    if distance_type.startswith('cosine_'):
        return distance_type[len('cosine_'):], 'cosine'
    elif distance_type == 'euclidean':
        return 'raw', 'l2'
    else:
        return distance_type, 'l2'


def _get_repr_features(inputs: np.ndarray, targets: np.ndarray,
                        repr_name: str) -> np.ndarray:
    """Get features for a named representation (raw, temporal, spatial)."""
    if repr_name == 'raw':
        return get_flat_features(inputs, targets)
    elif repr_name == 'temporal':
        return get_temporal_features(inputs, targets)
    elif repr_name == 'spatial':
        return get_spatial_features(inputs, targets)
    else:
        raise ValueError(f"Unknown representation: {repr_name}")


# ---------------------------------------------------------------------------
# Distance matrix computation
# ---------------------------------------------------------------------------

def _gpu_pairwise_l2(features: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Compute pairwise L2 distance matrix on GPU with batched computation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = features.shape[0]
    dist_mat = torch.empty((N, N), dtype=torch.float32, device='cpu')
    x_norm = np.sum(features ** 2, axis=1)

    for i in tqdm(range(0, N, batch_size), desc="Computing L2 distance"):
        end_i = min(i + batch_size, N)
        x_b = torch.from_numpy(features[i:end_i]).to(device).float()
        xn_b = torch.from_numpy(x_norm[i:end_i]).to(device).float().unsqueeze(1)

        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            x_r = torch.from_numpy(features[j:end_j]).to(device).float()
            xn_r = torch.from_numpy(x_norm[j:end_j]).to(device).float().unsqueeze(0)

            d2 = torch.clamp(xn_b + xn_r - 2 * (x_b @ x_r.T), min=0.0)
            dist_mat[i:end_i, j:end_j] = torch.sqrt(d2).cpu()

    return dist_mat.numpy()


def _gpu_pairwise_cosine_dist(features: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Compute pairwise cosine distance matrix: 1 - cos(a, b). Range [0, 2]."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = features.shape[0]

    # Pre-normalize to unit vectors
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = (features / norms).astype(np.float32)

    dist_mat = torch.empty((N, N), dtype=torch.float32, device='cpu')

    for i in tqdm(range(0, N, batch_size), desc="Computing cosine distance"):
        end_i = min(i + batch_size, N)
        x_b = torch.from_numpy(normalized[i:end_i]).to(device)

        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            x_r = torch.from_numpy(normalized[j:end_j]).to(device)

            cos_sim = x_b @ x_r.T
            dist_mat[i:end_i, j:end_j] = (1.0 - cos_sim).cpu()

    return dist_mat.numpy()


def compute_distance_matrix(inputs: np.ndarray, targets: np.ndarray,
                            distance_type: str = 'euclidean',
                            alpha: float = 0.5, beta: float = 0.5,
                            batch_size: int = 256) -> np.ndarray:
    """
    Compute pairwise distance matrix using specified distance type.

    Args:
        inputs:  (N, T_in, Nodes, F_forward)
        targets: (N, T_out, Nodes, F_target)
        distance_type: one of 8 types (see module docstring)
        alpha, beta: weights for combined distance (default 0.5 each)
        batch_size: GPU batch size for distance computation

    Returns:
        (N, N) distance matrix as numpy array
    """
    repr_name, metric = parse_distance_type(distance_type)
    dist_fn = _gpu_pairwise_cosine_dist if metric == 'cosine' else _gpu_pairwise_l2

    if repr_name == 'combined':
        d_t = dist_fn(get_temporal_features(inputs, targets), batch_size)
        d_s = dist_fn(get_spatial_features(inputs, targets), batch_size)
        # Normalize each to [0, 1] before combining
        t_max = d_t.max()
        s_max = d_s.max()
        if t_max > 0:
            d_t = d_t / t_max
        if s_max > 0:
            d_s = d_s / s_max
        return alpha * d_t + beta * d_s
    else:
        features = _get_repr_features(inputs, targets, repr_name)
        return dist_fn(features, batch_size)
