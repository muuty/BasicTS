"""
STMAE: Spatio-Temporal Masked Autoencoder for traffic forecasting.

This module implements a backbone-agnostic masked autoencoder that performs
self-supervised pre-training using:
1. Feature masking: Patches of time steps are masked and reconstructed
2. Structure masking: Edges in the graph are masked using random walks

Reference: Original implementation from STMAE paper
"""

from .arch import STMAE, STMAEForecaster
from .loss import STMAELoss, StructureLoss, FeatureLoss

__all__ = [
    'STMAE',
    'STMAEForecaster',
    'STMAELoss',
    'StructureLoss',
    'FeatureLoss',
]
