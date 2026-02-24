#!/usr/bin/env python3
"""FedAvg (Federated Averaging) Aggregator"""

from typing import Dict, Tuple, Optional
from collections import OrderedDict
import copy

import torch

from .base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg: Federated Averaging
    
    논문: Communication-Efficient Learning of Deep Networks from Decentralized Data
    링크: https://arxiv.org/abs/1602.05629
    
    각 클라이언트가 로컬 데이터로 여러 epoch 학습 후,
    서버가 데이터 크기 기반 가중 평균으로 글로벌 모델을 업데이트합니다.
    """
    
    def __init__(self):
        super().__init__()
    
    def aggregate(
        self,
        client_params: Dict[int, OrderedDict],
        client_weights: Dict[int, float],
        global_params: Optional[OrderedDict],
    ) -> Tuple[OrderedDict, Dict]:
        """
        FedAvg 집계: 가중 평균
        
        Args:
            client_params: {client_id: state_dict}
            client_weights: {client_id: weight} (데이터 크기 비율, 합이 1)
            global_params: 사용하지 않음 (FedProx에서 사용)
            
        Returns:
            averaged_params: 가중 평균된 파라미터
            comm_info: 통신 비용 정보
        """
        # 가중치 정규화 (합이 1이 되도록)
        total_weight = sum(client_weights.values())
        normalized_weights = {
            cid: w / total_weight for cid, w in client_weights.items()
        }
        
        # 첫 번째 클라이언트의 파라미터 구조 복사
        first_client_id = list(client_params.keys())[0]
        averaged_params = copy.deepcopy(client_params[first_client_id])
        
        # 가중 평균 계산
        for key in averaged_params.keys():
            # 노드별 파라미터는 스킵
            if self._is_node_specific_param(key):
                continue
            
            # 초기화
            averaged_params[key] = torch.zeros_like(
                averaged_params[key], 
                dtype=torch.float32
            )
            
            # 가중 합
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

