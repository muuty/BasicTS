#!/usr/bin/env python3
"""
Federated Learning Module

다양한 Federated Learning 전략을 basicts에서 구현합니다.
- FedAvg: Federated Averaging
- FedProx: Federated Optimization with Proximal Term

Usage:
    from baselines.FederatedLearning.arch import FederatedLearningModel
    from baselines.FederatedLearning.aggregators import get_aggregator
    from basicts.runners import FederatedLearningRunner
    
    model = FederatedLearningModel(
        base_model_class=STAEformer,
        base_model_params={...},
        num_clients=4,
        total_nodes=207,
        output_dim=1,
        aggregator_type='fedavg',  # or 'fedprox'
        aggregator_params={},  # {'mu': 0.01} for FedProx
    )
"""

from .arch import FederatedLearningModel
from .aggregators import get_aggregator, FedAvgAggregator, FedProxAggregator

__all__ = [
    "FederatedLearningModel",
    "get_aggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
]

