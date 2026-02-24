"""
ST-SSL: Self-Supervised Learning for Spatio-Temporal Traffic Prediction.

Paper: Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction
Link: https://arxiv.org/abs/2212.04475
Official Code: https://github.com/Echo-Ji/ST-SSL
Venue: AAAI 2023

This implementation supports:
1. Pre-training mode: Learn representations with contrastive losses
2. Fine-tuning mode: Predict traffic flow with pretrained encoder
3. End-to-end training: Joint training of all components

Key Components:
- STEncoder: Spatio-temporal encoder with Chebyshev graph convolutions
- SpatialHeteroModel: Spatial heterogeneity modeling with prototypes
- TemporalHeteroModel: Temporal heterogeneity modeling with discriminator
- Augmentations: Topology and traffic augmentation functions
"""
from .arch import STSSL, STSSLWrapper, STEncoder, MLP
from .arch import SpatialHeteroModel, TemporalHeteroModel
from .arch import sim_global, aug_topology, aug_traffic
from .loss import stssl_loss, STSSLLoss
from .runner import STSSLRunner

__all__ = [
    # Main model
    'STSSL',
    'STSSLWrapper',
    # Encoder components
    'STEncoder',
    'MLP',
    # Heterogeneity models
    'SpatialHeteroModel',
    'TemporalHeteroModel',
    # Augmentations
    'sim_global',
    'aug_topology',
    'aug_traffic',
    # Loss
    'stssl_loss',
    'STSSLLoss',
    # Runner
    'STSSLRunner',
]
