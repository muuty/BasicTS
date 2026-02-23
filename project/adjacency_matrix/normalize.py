"""
Model-specific adjacency normalization.

Takes a raw adjacency matrix and transforms it to the format
expected by each GNN model. Reuses existing functions from
basicts/utils/adjacent_matrix_norm.py.
"""

import numpy as np
import torch

from basicts.utils.adjacent_matrix_norm import (
    calculate_transition_matrix,
    calculate_symmetric_normalized_laplacian,
)
from contrastive.augmentation import BaseGSO


# Model → normalization function mapping
MODEL_NORMALIZERS = {}


def register_normalizer(model_name):
    """Register a normalization function for a model."""
    def decorator(fn):
        MODEL_NORMALIZERS[model_name] = fn
        return fn
    return decorator


def normalize_for_model(raw_adj, model_name):
    """Apply model-specific normalization to a raw adjacency matrix.

    Args:
        raw_adj: np.ndarray of shape (N, N), raw adjacency matrix.
        model_name: str, e.g. 'GWNet', 'MTGNN', 'DGCRN', 'STGCN'.

    Returns:
        Model-specific format (list of tensors, single tensor, etc.)
    """
    return MODEL_NORMALIZERS[model_name](raw_adj)


@register_normalizer("GWNet")
def _gwnet_normalize(raw_adj):
    """GWNet: doubletransition → list of 2 torch.Tensors.

    [D_out^{-1} A,  D_in^{-1} A^T]
    """
    fwd = calculate_transition_matrix(raw_adj).T  # forward diffusion
    bwd = calculate_transition_matrix(raw_adj.T).T  # backward diffusion
    return [
        torch.tensor(np.array(fwd), dtype=torch.float32),
        torch.tensor(np.array(bwd), dtype=torch.float32),
    ]


@register_normalizer("STGCN")
def _stgcn_normalize(raw_adj):
    """STGCN: normlap (I - D^{-1/2}AD^{-1/2}) wrapped in BaseGSO.

    STGCN uses Chebyshev spectral graph convolution on the symmetric
    normalized Laplacian. Returns a BaseGSO wrapper for interface
    compatibility with the STGCN architecture.
    """
    adj = np.array(raw_adj, dtype=np.float32)
    normlap = calculate_symmetric_normalized_laplacian(adj).astype(np.float32).todense()
    gso_tensor = torch.tensor(np.array(normlap), dtype=torch.float32)
    return BaseGSO(gso_tensor)


@register_normalizer("MTGNN")
def _mtgnn_normalize(raw_adj):
    """MTGNN: raw adjacency (no self-loops) → single torch.Tensor.

    MTGNN's mixprop internally adds self-loops (adj + I) and row-normalizes,
    so we provide the raw adjacency without self-loops. This guarantees
    row sums >= 1 after self-loop addition, avoiding division by zero.
    """
    adj = np.array(raw_adj, dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return torch.tensor(adj, dtype=torch.float32)
