#!/usr/bin/env python3
"""Federated Learning Aggregation Strategies"""

from .base import BaseAggregator
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator

__all__ = [
    "BaseAggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
]

# Aggregator registry for easy lookup
AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
    "fedprox": FedProxAggregator,
}


def get_aggregator(name: str, **kwargs) -> BaseAggregator:
    """Get aggregator by name.
    
    Args:
        name: Aggregator name ('fedavg', 'fedprox', etc.)
        **kwargs: Aggregator-specific parameters
        
    Returns:
        BaseAggregator instance
    """
    name = name.lower()
    if name not in AGGREGATOR_REGISTRY:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(AGGREGATOR_REGISTRY.keys())}")
    return AGGREGATOR_REGISTRY[name](**kwargs)

