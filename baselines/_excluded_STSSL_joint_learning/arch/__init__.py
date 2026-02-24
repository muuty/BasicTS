"""
ST-SSL Architecture Components.

ST-SSL: Self-Supervised Learning for Spatio-Temporal Traffic Prediction
Paper: https://arxiv.org/abs/2212.04475
"""
from .stssl_arch import STSSL, STSSLWrapper
from .encoder import STEncoder, MLP, TemporalConvLayer, SpatioConvLayer, Pooler
from .hetero_models import SpatialHeteroModel, TemporalHeteroModel
from .augmentations import sim_global, aug_topology, aug_traffic

__all__ = [
    'STSSL',
    'STSSLWrapper',
    'STEncoder',
    'MLP',
    'TemporalConvLayer',
    'SpatioConvLayer',
    'Pooler',
    'SpatialHeteroModel',
    'TemporalHeteroModel',
    'sim_global',
    'aug_topology',
    'aug_traffic',
]
