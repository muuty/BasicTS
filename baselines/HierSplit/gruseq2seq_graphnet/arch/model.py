#!/usr/bin/env python3
"""GRU Seq2Seq + GraphNet Hierarchical Split Learning 통합 모델"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .client import GRUSeq2SeqGraphNetHierClientModel
from .server import GRUSeq2SeqGraphNetHierServerModel


class GRUSeq2SeqGraphNetHierModel(nn.Module):
    def __init__(
        self,
        num_clients: int,
        client_nodes_list: List[List[int]],
        client_model_params: Dict,
        server_model_params: Dict,
        num_tokens: int,
        pooling_method: str,
        subgraph_adj_list: List[Optional[torch.Tensor]] = None,
        edge_index_list: List[Optional[torch.Tensor]] = None,
        edge_weight_list: List[Optional[torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.num_clients = num_clients
        self.client_nodes_list = client_nodes_list
        self.num_tokens = num_tokens
        self.pooling_method = pooling_method
        
        # 클라이언트 모델 생성
        self.client_models = nn.ModuleList()
        for client_id, client_nodes in enumerate(client_nodes_list):
            params = client_model_params.copy()

            params['num_nodes'] = len(client_nodes)
            params['num_tokens'] = num_tokens
            params['pooling_method'] = pooling_method
            params['subgraph_adj'] = subgraph_adj_list[client_id] if subgraph_adj_list else None
            params['edge_index'] = edge_index_list[client_id] if edge_index_list else None
            if edge_weight_list is not None:
                params['edge_attr'] = edge_weight_list[client_id]
            else:
                params['edge_attr'] = None
            
            self.client_models.append(GRUSeq2SeqGraphNetHierClientModel(**params))
        
        self.server_model = GRUSeq2SeqGraphNetHierServerModel(**server_model_params)
        
        self.total_forward_bytes = 0
        self.total_backward_bytes = 0
    
    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor] = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """
        history_data: (B, L_in, Total_N, C)
        """
        B, L_in, _, C = history_data.shape
        L_out = self.client_models[0].horizon
        
        # --- Phase 1: Client Encoding & Pooling ---
        client_h_encodes = {}
        client_graph_encodings = {}
        client_pooled = {}
        token_sizes = []
        
        for client_id, client_model in enumerate(self.client_models):
            nodes = self.client_nodes_list[client_id]
            client_history = history_data[:, :, nodes, :]
            
            # [Fix] edge_index 등을 여기서 명시적으로 넘기지 않아도, 
            # Client Model 내부 buffer를 사용하므로 안전함.
            h_encode, graph_encoding, pooled = client_model.forward_encoder_graphnet_pool(
                client_history
            )
            
            client_h_encodes[client_id] = h_encode
            client_graph_encodings[client_id] = graph_encoding
            client_pooled[client_id] = pooled
            token_sizes.append(pooled.shape[2]) # dim 2 is num_tokens
        
        # --- Independent Learning Mode (num_tokens=0) ---
        if self.num_tokens == 0:
            # Pooling과 Server를 건너뛰고 각 클라이언트가 독립적으로 예측
            total_nodes = history_data.shape[2]
            pred_all = torch.zeros(B, L_out, total_nodes, 1, device=history_data.device)
            
            for client_id, client_model in enumerate(self.client_models):
                nodes = self.client_nodes_list[client_id]
                client_history = history_data[:, :, nodes, :]
                client_future = future_data[:, :, nodes, :] if future_data is not None else None
                
                # Server 없이 독립적으로 예측 (server_features=None)
                pred = client_model.forward_decoder(
                    client_history,
                    client_h_encodes[client_id],
                    client_graph_encodings[client_id],
                    None,  # server_features=None
                    client_future,
                    batch_seen
                )
                
                pred_all[:, :, nodes, :] = pred
            
            # 통신량 없음 (Independent Learning)
            return {
                'prediction': pred_all,
                'all_pooled_features': None,
                'server_features': None,
                'forward_communication': {
                    'client_to_server_bytes': 0,
                    'server_to_client_bytes': 0
                }
            }
        
        # --- Phase 2: Server Aggregation (num_tokens > 0) ---
        pooled_list = [client_pooled[i] for i in range(self.num_clients)]
        
        # [Critical Fix] Concatenation Dimension
        # (B, Layers, Tokens, H) 형태이므로, Tokens(dim=2) 방향으로 합쳐야 함
        all_pooled = torch.cat(pooled_list, dim=2) 
        
        if train and torch.is_grad_enabled() and all_pooled.requires_grad:
            all_pooled.retain_grad()
            
        # Server Forward
        server_features = self.server_model(all_pooled) # (B, Layers, Total_Tokens, H)
        
        if train and torch.is_grad_enabled() and server_features.requires_grad:
            server_features.retain_grad()
            
        # --- Phase 3: Split & Client Decoding ---
        # dim=2 (Token dimension) 기준으로 다시 분할
        server_features_split = torch.split(server_features, token_sizes, dim=2)
        
        # 결과 담을 텐서 초기화 (Total_N 기준)
        total_nodes = history_data.shape[2]
        pred_all = torch.zeros(B, L_out, total_nodes, 1, device=history_data.device)
        
        for client_id, client_model in enumerate(self.client_models):
            nodes = self.client_nodes_list[client_id]
            
            server_features_client = server_features_split[client_id]
            
            client_history = history_data[:, :, nodes, :]
            client_future = future_data[:, :, nodes, :] if future_data is not None else None
            
            pred = client_model.forward_decoder(
                client_history,
                client_h_encodes[client_id],
                client_graph_encodings[client_id],
                server_features_client,
                client_future,
                batch_seen
            )
            
            pred_all[:, :, nodes, :] = pred
        
        # 통신량 계산 (Bytes)
        # Forward Pass: Client->Server (pooled features) + Server->Client (processed features)
        fwd_c2s = all_pooled.numel() * 4 
        fwd_s2c = server_features.numel() * 4
        
        if train:
            self.total_forward_bytes += (fwd_c2s + fwd_s2c)
            
        return {
            'prediction': pred_all,
            'all_pooled_features': all_pooled,
            'server_features': server_features,
            'forward_communication': {
                'client_to_server_bytes': fwd_c2s,
                'server_to_client_bytes': fwd_s2c
            }
        }
    
    def compute_backward_comm(self, forward_return: Dict) -> Dict[str, int]:
        """
        Backward 통신량 계산
        
        Hierarchical Split Learning에서 backward 시 gradient가 서버↔클라이언트 간에 전송됨
        - server → client: all_pooled_features의 grad
        - client → server: server_features의 grad
        
        Args:
            forward_return: forward() 반환값 (all_pooled_features, server_features 포함)
            
        Returns:
            Dict with 'client_to_server' and 'server_to_client' bytes
        """
        # Independent Learning 모드 (num_tokens=0)일 때는 통신량 없음
        if self.num_tokens == 0:
            return {'client_to_server': 0, 'server_to_client': 0}
        
        b_c2s = 0
        b_s2c = 0
        
        # 서버 → 클라이언트: pooled features의 gradient
        all_pooled = forward_return.get('all_pooled_features')
        if all_pooled is not None and all_pooled.grad is not None:
            b_s2c = all_pooled.grad.numel() * all_pooled.grad.element_size()
        
        # 클라이언트 → 서버: server features의 gradient
        server_features = forward_return.get('server_features')
        if server_features is not None and server_features.grad is not None:
            b_c2s = server_features.grad.numel() * server_features.grad.element_size()
        
        self.total_backward_bytes += (b_c2s + b_s2c)
        
        return {'client_to_server': b_c2s, 'server_to_client': b_s2c}
    
    def get_communication_stats(self) -> Dict[str, float]:
        """총 통신량 통계 반환 (MB 단위)"""
        return {
            'total_forward_mb': self.total_forward_bytes / (1024 * 1024),
            'total_backward_mb': self.total_backward_bytes / (1024 * 1024),
            'total_mb': (self.total_forward_bytes + self.total_backward_bytes) / (1024 * 1024),
        }
    
    def get_client_models(self) -> List[nn.Module]:
        """클라이언트 모델 리스트 반환"""
        return list(self.client_models)
    
    def get_server_model(self) -> nn.Module:
        """서버 모델 반환"""
        return self.server_model

