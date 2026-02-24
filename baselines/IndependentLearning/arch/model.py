#!/usr/bin/env python3
"""
Independent Learning Model Wrapper

여러 독립적인 클라이언트 모델을 하나의 nn.Module로 감싸서
basicts의 기존 인프라와 호환되게 합니다.

Usage:
    from baselines.STAEformer.arch import STAEformer
    
    model = IndependentLearningModel(
        base_model_class=STAEformer,
        base_model_params={...},
        num_clients=4,
        total_nodes=207,
        partition_method='random',  # or 'metis', 'spectral'
        adj_matrix=adj_mx,  # optional, for graph-based partitioning
    )
"""

from typing import Dict, List, Type, Optional
import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../../.."))

from baselines.node_partition import setup_split_learning_nodes


class IndependentLearningModel(nn.Module):
    """
    Independent Learning을 위한 모델 래퍼
    
    각 클라이언트가 자신의 노드 서브셋에 대해 독립적인 모델을 가지고,
    forward 시 각 클라이언트 모델이 해당 노드만 처리합니다.
    """
    
    def __init__(
        self,
        base_model_class: Type[nn.Module],
        base_model_params: Dict,
        num_clients: int,
        total_nodes: int,
        output_dim: int,
        partition_method: Optional[str],
        adj_matrix: Optional[torch.Tensor],
        metadata: Optional[Dict[str, List[str]]],
    ):
        """
        Args:
            base_model_class: 베이스 모델 클래스 (e.g., STAEformer)
            base_model_params: 베이스 모델 파라미터 (num_nodes 제외)
            num_clients: 클라이언트 수
            total_nodes: 전체 노드 수
            output_dim: 출력 차원
            partition_method: 노드 분할 방법 ('random', 'metis', 'spectral'), client_nodes_list 제공 시 무시
            adj_matrix: 인접 행렬 (graph-based partitioning용, optional)
            client_nodes_list: 미리 계산된 클라이언트별 노드 리스트 (제공 시 partition_method 무시)
            subgraph_adj_list: 각 클라이언트의 subgraph adjacency matrix 리스트 (HierSplit 방식, optional)
        """
        super().__init__()
        
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.num_clients = num_clients
        self.total_nodes = total_nodes
        self.partition_method = partition_method
        self.output_dim = output_dim
        
        # 노드 분할: client_nodes_list가 제공되면 그대로 사용, 아니면 partition_method로 분할
        self.client_nodes_list, self.subgraph_adj_list, _ = setup_split_learning_nodes(
            num_nodes=self.total_nodes,
            num_clients=self.num_clients,
            grouping_method=self.partition_method,
            adj_matrix=adj_matrix,
            metadata=metadata,
        )
        
        if self.num_clients != len(self.client_nodes_list):
            print(f"num_clients ({self.num_clients}) != len(client_nodes_list) ({len(self.client_nodes_list)}). Setting num_clients to {len(self.client_nodes_list)}")
            self.num_clients = len(self.client_nodes_list)
        
        # supports 처리 (STGformer 등에서 사용)
        # base_model_params에 'supports' 키가 있으면 supports를 사용하는 모델 (STGformer)
        uses_supports = 'supports' in base_model_params
        supports = base_model_params.get('supports', None) if uses_supports else None
        
        # edge_index, edge_attr 처리 (GRUSeq2SeqWithGraphNet 등에서 사용)
        # base_model_params에 'edge_index' 또는 'edge_attr' 키가 있으면 edge를 사용하는 모델
        uses_edges = 'edge_index' in base_model_params or 'edge_attr' in base_model_params
        
        # 클라이언트 모델들 생성
        self.client_models = nn.ModuleList()
        for client_idx, client_nodes in enumerate(self.client_nodes_list):
            params = base_model_params.copy()
            params['num_nodes'] = len(client_nodes)
            params['adj_matrix'] = self.subgraph_adj_list[client_idx]
            print(params)
            
            # supports를 사용하는 모델(STGformer)에만 supports 처리
            if uses_supports:
                # HierSplit 방식: subgraph_adj_list에서 supports 생성
                if self.subgraph_adj_list is not None and len(self.subgraph_adj_list) > client_idx:
                    adj = self.subgraph_adj_list[client_idx]
                    
                    # supports가 명시되지 않았거나 None일 경우, subgraph_adj_list에서 가져와서 넣어줌
                    if params.get("supports") is None:
                        params["supports"] = [adj]
                else:
                    # subgraph_adj_list가 없으면 기존 방식 사용 (supports를 클라이언트별로 분할)
                    if supports is not None and isinstance(supports, list) and len(supports) > 0:
                        # supports를 클라이언트별로 분할
                        client_supports = self._partition_supports_for_client(supports, client_nodes)
                        params['supports'] = client_supports
                    elif params.get("supports") is None:
                        params["supports"] = None
            else:
                # supports를 사용하지 않는 모델(STAEformer, GRUSeq2SeqWithGraphNet 등)은 supports 제거
                params.pop('supports', None)
            
            # edge_index, edge_attr 처리 (GRUSeq2SeqWithGraphNet 등)
            if uses_edges:
                # subgraph_adj_list가 있으면 그것을 사용해서 edge_index, edge_attr 생성
                if self.subgraph_adj_list is not None and len(self.subgraph_adj_list) > client_idx:
                    adj = self.subgraph_adj_list[client_idx]
                    edge_index, edge_attr = self._adj_to_edge_index(adj)
                    params['edge_index'] = edge_index
                    params['edge_attr'] = edge_attr
                elif adj_matrix is not None:
                    # adj_matrix가 있으면 클라이언트 노드에 해당하는 subgraph 추출
                    client_adj = adj_matrix[client_nodes][:, client_nodes]
                    edge_index, edge_attr = self._adj_to_edge_index(client_adj)
                    params['edge_index'] = edge_index
                    params['edge_attr'] = edge_attr
                else:
                    # adj_matrix가 없으면 edge_index, edge_attr를 None으로 유지
                    if params.get('edge_index') is None:
                        params['edge_index'] = None
                    if params.get('edge_attr') is None:
                        params['edge_attr'] = None
            
            client_model = base_model_class(**params)
            self.client_models.append(client_model)
        
        # 노드 인덱스를 버퍼로 등록 (GPU 이동 시 함께 이동)
        for client_idx, client_nodes in enumerate(self.client_nodes_list):
            self.register_buffer(
                f'client_nodes_{client_idx}',
                torch.tensor(client_nodes, dtype=torch.long)
            )
        
        self._print_init_info()
    

    def _partition_supports_for_client(self, supports: List[torch.Tensor], client_nodes: List[int]) -> List[torch.Tensor]:
        """
        supports를 특정 클라이언트의 서브그래프로 분할
        
        Args:
            supports: 전체 그래프의 supports 리스트 (각 요소는 [N, N] 형태)
            client_nodes: 클라이언트 노드 인덱스 리스트
            
        Returns:
            클라이언트용 supports 리스트
        """
        client_supports = []
        for support in supports:
            # 클라이언트 노드에 해당하는 서브그래프 추출
            if isinstance(support, torch.Tensor):
                client_support = support[client_nodes][:, client_nodes]
            else:
                # numpy array인 경우
                import numpy as np
                if isinstance(support, np.ndarray):
                    client_support = torch.from_numpy(support[client_nodes][:, client_nodes])
                else:
                    client_support = support
            client_supports.append(client_support)
        
        return client_supports
    
    def _adj_to_edge_index(self, adj: torch.Tensor) -> tuple:
        """
        Adjacency matrix를 edge_index와 edge_attr로 변환
        
        Args:
            adj: Adjacency matrix [N, N]
            
        Returns:
            tuple: (edge_index [2, E], edge_attr [E])
        """
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float()
        
        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
        edge_attr = adj[adj > 0]
        
        return edge_index, edge_attr
    
    def _print_init_info(self):
        """초기화 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n🎯 IndependentLearningModel 초기화")
        print(f"  - 베이스 모델: {self.base_model_class.__name__}")
        print(f"  - 클라이언트 수: {self.num_clients}")
        print(f"  - 전체 노드 수: {self.total_nodes}")
        print(f"  - 분할 방법: {self.partition_method}")
        print(f"  - 총 파라미터: {total_params:,}")
        for i, nodes in enumerate(self.client_nodes_list):
            print(f"  - Client {i}: {len(nodes)} nodes")
    
    def get_client_nodes(self, client_idx: int) -> torch.Tensor:
        """클라이언트별 노드 인덱스 반환"""
        return getattr(self, f'client_nodes_{client_idx}')
    
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
        Forward pass
        
        Args:
            history_data: [B, L_in, N, C] 입력 데이터
            future_data: [B, L_out, N, C] 미래 데이터 (optional)
            batch_seen: 현재까지 본 배치 수
            epoch: 현재 에포크
            train: 학습 모드 여부
            
        Returns:
            Dict containing:
                - 'prediction': [B, L_out, N, output_dim] 예측값
                - 'client_nodes_list': 각 클라이언트의 노드 인덱스 리스트
        """
        B, L_in, N, C = history_data.shape
        device = history_data.device
        dtype = history_data.dtype
        
        # 출력 길이 결정
        if future_data is not None:
            L_out = future_data.shape[1]
        else:
            L_out = self.base_model_params.get('out_steps', L_in)
        
        # 전체 노드 크기의 prediction 텐서 초기화
        pred_all = torch.zeros(B, L_out, N, self.output_dim, device=device, dtype=dtype)
        
        # 각 클라이언트별로 forward
        for client_idx, client_model in enumerate(self.client_models):
            import json
            log_path = "/data/basicts/.cursor/debug.log"
            
            client_nodes = self.get_client_nodes(client_idx)
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "model.py:258",
                    "message": "client forward entry",
                    "data": {
                        "client_idx": client_idx,
                        "client_nodes_type": str(type(client_nodes)),
                        "client_nodes_shape": list(client_nodes.shape) if hasattr(client_nodes, 'shape') else None,
                        "client_nodes_min": int(client_nodes.min().item()) if hasattr(client_nodes, 'min') else None,
                        "client_nodes_max": int(client_nodes.max().item()) if hasattr(client_nodes, 'max') else None,
                        "history_data_shape": list(history_data.shape),
                        "history_data_node_dim": history_data.shape[2],
                        "out_of_range": [int(n.item()) for n in client_nodes if n.item() < 0 or n.item() >= history_data.shape[2]] if hasattr(client_nodes, '__iter__') else []
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion
            
            # 해당 노드만 슬라이스
            x_client = history_data[:, :, client_nodes, :]
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "model.py:261",
                    "message": "after slicing",
                    "data": {
                        "client_idx": client_idx,
                        "x_client_shape": list(x_client.shape),
                        "expected_nodes": len(client_nodes) if hasattr(client_nodes, '__len__') else None,
                        "client_model_num_nodes": getattr(client_model, 'num_nodes', None)
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion
            
            future_client = future_data[:, :, client_nodes, :] if future_data is not None else None
            
            # 클라이언트 모델 forward
            pred_client = client_model(
                history_data=x_client,
                future_data=future_client,
                batch_seen=batch_seen,
                epoch=epoch,
                train=train,
                **kwargs
            )
            
            # 텐서 또는 딕셔너리 처리
            if isinstance(pred_client, dict):
                pred_client = pred_client['prediction']
            
            # 전체 예측에 할당
            pred_all[:, :, client_nodes, :] = pred_client
        
        return {
            'prediction': pred_all,
            'client_nodes_list': self.client_nodes_list,
        }


    def get_client_nodes_list(self) -> List[List[int]]:
        """클라이언트별 노드 인덱스 리스트 반환"""
        return self.client_nodes_list