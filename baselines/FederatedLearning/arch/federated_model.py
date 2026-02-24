#!/usr/bin/env python3
"""
Federated Learning Model Wrapper

여러 독립적인 클라이언트 모델을 관리하고,
Federated Learning의 local training과 aggregation을 지원합니다.

Usage:
    from baselines.STAEformer.arch import STAEformer
    from baselines.FederatedLearning.aggregators import get_aggregator
    
    model = FederatedLearningModel(
        base_model_class=STAEformer,
        base_model_params={...},
        num_clients=4,
        total_nodes=207,
        output_dim=1,
        aggregator_type='fedavg',
        aggregator_params={},
    )
"""

from typing import Dict, List, Type, Optional
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../../.."))

from ..aggregators import get_aggregator, BaseAggregator
from baselines.node_partition import random_partition, metis_partition, spectral_partition


class FederatedLearningModel(nn.Module):
    """
    Federated Learning을 위한 모델 래퍼
    
    각 클라이언트가 독립적인 모델을 가지고,
    local training 후 aggregation으로 동기화합니다.
    """
    
    def __init__(
        self,
        base_model_class: Type[nn.Module],
        base_model_params: Dict,
        num_clients: int,
        total_nodes: int,
        output_dim: int,
        aggregator_type: str,
        aggregator_params: Dict,
        partition_method: Optional[str],
        adj_matrix: Optional[torch.Tensor],
        client_nodes_list: Optional[List[List[int]]],
    ):
        """
        Args:
            base_model_class: 베이스 모델 클래스 (e.g., STAEformer)
            base_model_params: 베이스 모델 파라미터 (num_nodes 제외)
            num_clients: 클라이언트 수
            total_nodes: 전체 노드 수
            output_dim: 출력 차원
            aggregator_type: Aggregation 전략 ('fedavg', 'fedprox')
            aggregator_params: Aggregator 파라미터 (e.g., {'mu': 0.01} for FedProx)
            partition_method: 노드 분할 방법 ('random', 'metis', 'spectral')
            adj_matrix: 인접 행렬 (graph-based partitioning용)
            client_nodes_list: 미리 계산된 노드 분할 (제공 시 partition_method 무시)
        """
        super().__init__()
        
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.num_clients = num_clients
        self.total_nodes = total_nodes
        self.output_dim = output_dim
        self.partition_method = partition_method
        
        # Aggregator 생성
        self.aggregator: BaseAggregator = get_aggregator(
            aggregator_type, **aggregator_params
        )
        self.aggregator_type = aggregator_type
        
        # 노드 분할
        if client_nodes_list is not None:
            self.client_nodes_list = client_nodes_list
        elif partition_method == 'random':
            self.client_nodes_list = random_partition(total_nodes, num_clients)
        elif partition_method == 'metis':
            self.client_nodes_list = metis_partition(adj_matrix)
        elif partition_method == 'spectral':
            self.client_nodes_list = spectral_partition(adj_matrix)
        else:
            raise ValueError(f"Unknown partition method: {partition_method}")
        # 클라이언트 모델들 생성
        self.client_models = nn.ModuleList()
        for client_idx, client_nodes in enumerate(self.client_nodes_list):
            params = base_model_params.copy()
            params['num_nodes'] = len(client_nodes)
            client_model = base_model_class(**params)
            self.client_models.append(client_model)
        
        # 노드 인덱스를 버퍼로 등록
        for client_idx, client_nodes in enumerate(self.client_nodes_list):
            self.register_buffer(
                f'client_nodes_{client_idx}',
                torch.tensor(client_nodes, dtype=torch.long)
            )
        
        # 글로벌 파라미터 저장 (aggregation 후, FedProx에서 사용)
        self._global_params: Optional[OrderedDict] = None
        
        self._print_init_info()
    

    
    def get_client_nodes(self, client_idx: int) -> torch.Tensor:
        """클라이언트별 노드 인덱스 반환"""
        return getattr(self, f'client_nodes_{client_idx}')
    
    def get_client_params(self, client_idx: int) -> OrderedDict:
        """클라이언트 모델 파라미터 반환"""
        return copy.deepcopy(self.client_models[client_idx].state_dict())
    
    def set_client_params(self, client_idx: int, params: OrderedDict):
        """클라이언트 모델 파라미터 설정"""
        self.client_models[client_idx].load_state_dict(params, strict=False)
    
    def get_all_client_params(self) -> Dict[int, OrderedDict]:
        """모든 클라이언트 파라미터 반환"""
        return {
            i: self.get_client_params(i) 
            for i in range(self.num_clients)
        }
    
    def get_client_data_weights(self) -> Dict[int, float]:
        """클라이언트별 데이터 가중치 (노드 수 기반)"""
        total_nodes = sum(len(nodes) for nodes in self.client_nodes_list)
        return {
            i: len(self.client_nodes_list[i]) / total_nodes
            for i in range(self.num_clients)
        }
    
    def aggregate_and_distribute(self) -> Dict:
        """
        모든 클라이언트 파라미터를 집계하고 배포합니다.
        
        Returns:
            comm_info: 통신 비용 정보
        """
        # 모든 클라이언트 파라미터 수집
        client_params = self.get_all_client_params()
        client_weights = self.get_client_data_weights()
        
        # Aggregation
        averaged_params, comm_info = self.aggregator.aggregate(
            client_params=client_params,
            client_weights=client_weights,
            global_params=self._global_params,
        )
        
        # 글로벌 파라미터 저장
        self._global_params = copy.deepcopy(averaged_params)
        
        # 모든 클라이언트에 배포
        for client_idx in range(self.num_clients):
            current_params = self.get_client_params(client_idx)
            
            # 노드별 파라미터는 유지하고, 나머지만 업데이트
            for key in averaged_params.keys():
                if self.aggregator._is_node_specific_param(key):
                    continue
                if key in current_params and current_params[key].shape == averaged_params[key].shape:
                    current_params[key] = averaged_params[key].clone()
            
            self.set_client_params(client_idx, current_params)
        
        return comm_info
    
    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor],
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs
    ) -> Dict:
        """
        Forward pass (evaluation용)
        
        모든 클라이언트가 해당 노드의 예측을 수행하고 합칩니다.
        """
        B, L_in, N, C = history_data.shape
        device = history_data.device
        dtype = history_data.dtype
        
        if future_data is not None:
            L_out = future_data.shape[1]
        else:
            L_out = self.base_model_params.get('out_steps', L_in)
        
        pred_all = torch.zeros(B, L_out, N, self.output_dim, device=device, dtype=dtype)
        
        for client_idx, client_model in enumerate(self.client_models):
            client_nodes = self.get_client_nodes(client_idx)
            
            x_client = history_data[:, :, client_nodes, :]
            future_client = future_data[:, :, client_nodes, :] if future_data is not None else None
            
            pred_client = client_model(
                history_data=x_client,
                future_data=future_client,
                batch_seen=batch_seen,
                epoch=epoch,
                train=train,
                **kwargs
            )
            
            if isinstance(pred_client, dict):
                pred_client = pred_client['prediction']
            
            pred_all[:, :, client_nodes, :] = pred_client
        
        return {
            'prediction': pred_all,
            'client_nodes_list': self.client_nodes_list,
        }


    def _print_init_info(self):
        """초기화 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n🌐 FederatedLearningModel 초기화")
        print(f"  - 베이스 모델: {self.base_model_class.__name__}")
        print(f"  - Aggregator: {self.aggregator_type}")
        print(f"  - 클라이언트 수: {self.num_clients}")
        print(f"  - 전체 노드 수: {self.total_nodes}")
        print(f"  - 분할 방법: {self.partition_method}")
        print(f"  - 총 파라미터: {total_params:,}")
        for i, nodes in enumerate(self.client_nodes_list):
            print(f"  - Client {i}: {len(nodes)} nodes")