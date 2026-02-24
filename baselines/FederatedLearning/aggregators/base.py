#!/usr/bin/env python3
"""Base Aggregator for Federated Learning"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import copy

import torch


class BaseAggregator(ABC):
    """
    Federated Learning Aggregator 기본 클래스
    
    다양한 aggregation 전략을 구현할 수 있는 인터페이스를 제공합니다.
    - FedAvg: 가중 평균
    - FedProx: 가중 평균 + proximal regularization
    - FedSGD: 단일 배치 학습 후 평균
    """
    
    # 노드 수에 따라 달라지는 파라미터 (aggregation에서 제외)
    NODE_SPECIFIC_PARAMS = ['adaptive_embedding', 'node_emb']
    
    def __init__(self):
        self.total_communication_bytes = 0
    
    @abstractmethod
    def aggregate(
        self,
        client_params: Dict[int, OrderedDict],
        client_weights: Dict[int, float],
        global_params: Optional[OrderedDict],
    ) -> Tuple[OrderedDict, Dict]:
        """
        클라이언트 파라미터를 집계합니다.
        
        Args:
            client_params: {client_id: state_dict} 각 클라이언트의 모델 파라미터
            client_weights: {client_id: weight} 각 클라이언트의 가중치 (보통 데이터 크기 비율)
            global_params: 이전 글로벌 파라미터 (FedProx 등에서 사용)
            
        Returns:
            aggregated_params: 집계된 파라미터
            comm_info: 통신 비용 정보
        """
        pass
    
    def get_proximal_loss(
        self,
        model: torch.nn.Module,
        global_params: OrderedDict,
    ) -> torch.Tensor:
        """
        Proximal regularization term 계산 (FedProx용)
        
        Args:
            model: 현재 클라이언트 모델
            global_params: 글로벌 파라미터
            
        Returns:
            proximal loss term
        """
        return torch.tensor(0.0)
    
    def _is_node_specific_param(self, key: str) -> bool:
        """노드 수에 따라 달라지는 파라미터인지 확인"""
        return any(skip in key for skip in self.NODE_SPECIFIC_PARAMS)
    
    def _compute_communication_bytes(
        self,
        params: OrderedDict,
        num_clients: int,
        direction: str,
    ) -> int:
        """
        통신 비용 계산
        
        Args:
            params: 전송할 파라미터
            num_clients: 클라이언트 수
            direction: 'upload' (client→server) or 'download' (server→client)
            
        Returns:
            bytes
        """
        total_bytes = 0
        for key, value in params.items():
            if not self._is_node_specific_param(key):
                total_bytes += value.numel() * value.element_size()
        
        if direction == 'upload':
            return total_bytes * num_clients
        else:  # download
            return total_bytes * num_clients
    
    def filter_aggregatable_params(
        self,
        params: OrderedDict,
        reference_params: OrderedDict,
    ) -> OrderedDict:
        """
        Aggregation 가능한 파라미터만 필터링
        (노드별 파라미터 제외, shape 일치하는 것만)
        
        Args:
            params: 필터링할 파라미터
            reference_params: 참조 파라미터 (shape 비교용)
            
        Returns:
            필터링된 파라미터
        """
        filtered = OrderedDict()
        for key, value in params.items():
            if self._is_node_specific_param(key):
                continue
            if key in reference_params and reference_params[key].shape == value.shape:
                filtered[key] = value
        return filtered

