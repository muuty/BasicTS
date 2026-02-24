#!/usr/bin/env python3
"""FedProx Aggregator"""

from typing import Dict, Tuple, Optional
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

from .base import BaseAggregator


class FedProxAggregator(BaseAggregator):
    """
    FedProx: Federated Optimization in Heterogeneous Networks
    
    논문: Federated Optimization in Heterogeneous Networks
    링크: https://arxiv.org/abs/1812.06127
    
    FedAvg에 proximal term을 추가하여 클라이언트 드리프트를 방지합니다.
    
    Local objective:
        h_k(w; w^t) = F_k(w) + (mu/2) * ||w - w^t||^2
        
    여기서 mu는 proximal coefficient, w^t는 글로벌 모델 파라미터입니다.
    """
    
    def __init__(self, mu: float):
        """
        Args:
            mu: Proximal coefficient (0.0이면 FedAvg와 동일)
        """
        super().__init__()
        self.mu = mu
    
    def aggregate(
        self,
        client_params: Dict[int, OrderedDict],
        client_weights: Dict[int, float],
        global_params: Optional[OrderedDict],
    ) -> Tuple[OrderedDict, Dict]:
        """
        FedProx 집계: FedAvg와 동일 (aggregation 자체는 같음)
        
        Proximal term은 local training에서 적용됨
        """
        # 가중치 정규화
        total_weight = sum(client_weights.values())
        normalized_weights = {
            cid: w / total_weight for cid, w in client_weights.items()
        }
        
        # 첫 번째 클라이언트의 파라미터 구조 복사
        first_client_id = list(client_params.keys())[0]
        averaged_params = copy.deepcopy(client_params[first_client_id])
        
        # 가중 평균 계산 (FedAvg와 동일)
        for key in averaged_params.keys():
            if self._is_node_specific_param(key):
                continue
            
            averaged_params[key] = torch.zeros_like(
                averaged_params[key],
                dtype=torch.float32
            )
            
            for client_id, params in client_params.items():
                if key in params and params[key].shape == averaged_params[key].shape:
                    weight = normalized_weights[client_id]
                    averaged_params[key] += weight * params[key].float()
        
        # 통신 비용 계산
        num_clients = len(client_params)
        upload_bytes = self._compute_communication_bytes(
            averaged_params, num_clients, 'upload'
        )
        download_bytes = self._compute_communication_bytes(
            averaged_params, num_clients, 'download'
        )
        
        comm_info = {
            'client_to_server_bytes': upload_bytes,
            'server_to_client_bytes': download_bytes,
            'total_communication_bytes': upload_bytes + download_bytes,
        }
        
        self.total_communication_bytes += comm_info['total_communication_bytes']
        
        return averaged_params, comm_info
    
    def get_proximal_loss(
        self,
        model: nn.Module,
        global_params: OrderedDict,
    ) -> torch.Tensor:
        """
        Proximal regularization term 계산
        
        L_prox = (mu/2) * sum(||w - w_global||^2)
        
        Args:
            model: 현재 클라이언트 모델
            global_params: 글로벌 파라미터
            
        Returns:
            proximal loss term
        """
        if self.mu == 0.0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        proximal_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_node_specific_param(name):
                continue
            if name in global_params:
                global_param = global_params[name].to(param.device)
                if global_param.shape == param.shape:
                    proximal_loss += torch.sum((param - global_param) ** 2)
        
        return (self.mu / 2.0) * proximal_loss

