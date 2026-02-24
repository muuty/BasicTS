#!/usr/bin/env python3
"""
Independent Learning Module

각 클라이언트가 자신의 노드 서브셋에 대해 독립적인 모델을 학습하는
Independent Learning을 basicts에서 구현합니다.

Usage:
    from baselines.IndependentLearning.arch import IndependentLearningModel
    from basicts.runners import IndependentLearningRunner
    
    model = IndependentLearningModel(
        base_model_class=STAEformer,
        base_model_params={...},
        num_clients=4,
        total_nodes=207,
        partition_method='random',
        adj_matrix=adj_mx,
        output_dim=1,
    )
"""

from .arch import IndependentLearningModel

__all__ = ["IndependentLearningModel"]

