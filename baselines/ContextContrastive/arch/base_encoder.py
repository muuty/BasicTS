"""
Base Representation Encoder Interface.

All representation encoders implement this interface, enabling:
1. Unified runner that works with any encoder type
2. Consistent checkpoint save/load
3. Easy swapping of encoder architectures
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseRepresentationEncoder(nn.Module, ABC):
    """
    Base class for all representation encoders.

    All encoders transform input time series into learned representations:
        Input:  [B, T, N, D] - batch, time, nodes, input_dim
        Output: [B, T, N, H] - batch, time, nodes, hidden_dim

    Encoder owns its adjacency matrix (if needed) via register_buffer.
    Subclasses must implement the `encode` method.
    """

    def __init__(self, input_dim: int, d_model: int, adj_path: str = None, **kwargs):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Output representation dimension
            adj_path: Path to adjacency matrix pickle file (optional)
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Load adjacency matrix if specified (encoder owns it)
        if adj_path:
            self._load_adjacency(adj_path)
        else:
            self.adj = None

    def _load_adjacency(self, adj_path: str):
        """Load adjacency matrix and store as buffer."""
        with open(adj_path, 'rb') as f:
            data = pickle.load(f)

        # Standard format: (sensor_ids, sensor_id_to_ind, adj_mx) or just array
        adj_mx = data[2] if hasattr(data, '__getitem__') and not hasattr(data, 'shape') else data

        # Handle sparse matrices
        if sp.issparse(adj_mx):
            adj_mx = adj_mx.toarray()

        self.register_buffer('adj', torch.from_numpy(np.asarray(adj_mx)).float())

    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode input time series into representations.

        Args:
            x: Input tensor [B, T, N, D]
            **kwargs: Additional encoder-specific arguments

        Returns:
            embedding: Encoded representations [B, T, N, H]
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls encode()."""
        return self.encode(x, **kwargs)

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder."""
        return self.d_model

    def get_config(self) -> Dict[str, Any]:
        """Return encoder configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'd_model': self.d_model,
        }

    def load_pretrained_weights(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> tuple:
        """
        Load pretrained weights from a checkpoint state_dict.

        Each encoder subclass can override this to handle its own weight mapping
        from different checkpoint formats.

        Args:
            state_dict: Full checkpoint state_dict (e.g., model_state_dict from pretrain)
            strict: Whether to strictly enforce key matching

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Default: extract keys with 'encoder.' prefix
        encoder_state_dict = self._extract_encoder_weights(state_dict)

        if not encoder_state_dict:
            # Fallback: try loading directly
            encoder_state_dict = state_dict

        return self.load_state_dict(encoder_state_dict, strict=strict)

    def _extract_encoder_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract encoder weights from a full model state_dict.

        Override this method in subclasses to handle different checkpoint formats.

        Args:
            state_dict: Full checkpoint state_dict

        Returns:
            Dictionary of encoder-only weights with appropriate key names
        """
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
                encoder_state_dict[new_key] = value
        return encoder_state_dict


# Registry for encoder types
ENCODER_REGISTRY: Dict[str, type] = {}


def register_encoder(name: str):
    """Decorator to register an encoder class."""
    def decorator(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def build_encoder(encoder_cfg: Dict[str, Any]) -> BaseRepresentationEncoder:
    """
    Build encoder from configuration.

    Args:
        encoder_cfg: Dictionary with 'type' and encoder-specific parameters

    Returns:
        Instantiated encoder
    """
    encoder_type = encoder_cfg.get('type', 'TransformerEncoder')

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )

    encoder_cls = ENCODER_REGISTRY[encoder_type]

    # Remove 'type' from params
    params = {k: v for k, v in encoder_cfg.items() if k != 'type'}

    return encoder_cls(**params)
