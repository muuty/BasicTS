#!/usr/bin/env python3
"""GRU Seq2Seq + GraphNet Split Learning 통합 모델"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .client import GRUSeq2SeqSplitClientModel
from .server import GraphNetSplitServerModel


class GRUSeq2SeqGraphNetSplitModel(nn.Module):
    def __init__(
        self,
        num_clients: int,
        client_nodes_list: List[List[int]],
        client_model_params: Dict,
        server_model_params: Dict,
    ):
        super().__init__()
        
        self.num_clients = num_clients
        self.client_nodes_list = client_nodes_list
        
        # 클라이언트 모델들 생성
        self.client_models = nn.ModuleList()
        for client_id, client_nodes in enumerate(client_nodes_list):
            params = client_model_params.copy()
            params['num_nodes'] = len(client_nodes)
            
            client_model = GRUSeq2SeqSplitClientModel(**params)
            self.client_models.append(client_model)
        
        # 서버 모델 생성
        self.server_model = GraphNetSplitServerModel(**server_model_params)
        
        # 통신 비용 추적
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
        B, L_in, N, C = history_data.shape
        L_out = self.client_models[0].horizon
        device = history_data.device
        
        # -------------------------------------------------------------
        # Phase 1: 클라이언트 인코더 수행
        # -------------------------------------------------------------
        client_h_encodes = {}
        
        for client_id in range(self.num_clients):
            client_model = self.client_models[client_id]
            nodes = self.client_nodes_list[client_id]
            client_history = history_data[:, :, nodes, :]
            
            # h_encode: (num_layers, B*N_sub, H)
            h_encode = client_model.forward_encoder(client_history)
            client_h_encodes[client_id] = h_encode
        
        # -------------------------------------------------------------
        # Phase 2: Pre-allocate & Assign (서버로 전송)
        # -------------------------------------------------------------
        # 차원 정보 추출
        sample_h = client_h_encodes[0]
        num_layers = sample_h.shape[0]
        hidden_size = sample_h.shape[2]
        
        # 전체 텐서 미리 생성 (Layers, B*N_total, H)
        # Reshape을 위해 (Layers, B, N_total, H)로 먼저 생성
        all_h_temp = torch.zeros(num_layers, B, N, hidden_size, device=device)
        
        for client_id in range(self.num_clients):
            h = client_h_encodes[client_id] # (Layers, B*N_sub, H)
            nodes = self.client_nodes_list[client_id]
            
            # Reshape local h to match grid: (Layers, B, N_sub, H)
            h_reshaped = h.view(num_layers, B, -1, hidden_size)
            
            # [Assign Logic] 제자리에 넣기
            all_h_temp[:, :, nodes, :] = h_reshaped
            
        # Flatten for Server input: (Layers, B*N, H)
        all_h_flat = all_h_temp.view(num_layers, B * N, hidden_size)
        
        fwd_c2s = all_h_flat.numel() * 4
        
        if train and torch.is_grad_enabled():
            if all_h_flat.requires_grad:
                all_h_flat.retain_grad()
        
        # -------------------------------------------------------------
        # Phase 3: 서버 GraphNet 처리
        # -------------------------------------------------------------
        edge_index = kwargs.get('edge_index', None)
        edge_attr = kwargs.get('edge_attr', None) # or edge_weight
        
        # graph_encoding: (Layers, B*N, H) - 이미 Global index(0~N) 순서임
        graph_encoding = self.server_model(all_h_flat, edge_index, edge_attr)
        
        fwd_s2c = graph_encoding.numel() * 4
        
        if train and torch.is_grad_enabled():
            if graph_encoding.requires_grad:
                graph_encoding.retain_grad()
        
        # -------------------------------------------------------------
        # Phase 4: 클라이언트 디코더 수행 (Slice by Index)
        # -------------------------------------------------------------
        # Reshape for slicing: (Layers, B, N, H)
        graph_encoding_reshaped = graph_encoding.view(num_layers, B, N, hidden_size)
        
        pred_all = torch.zeros(B, L_out, N, self.client_models[0].output_dim, device=device)
        
        for client_id in range(self.num_clients):
            client_model = self.client_models[client_id]
            nodes = self.client_nodes_list[client_id]
            
            # [Slice Logic] 인덱스로 바로 추출
            client_graph_enc = graph_encoding_reshaped[:, :, nodes, :]
            # Flatten back for Decoder: (Layers, B*N_sub, H)
            client_graph_enc = client_graph_enc.reshape(num_layers, B * len(nodes), hidden_size)
            
            # Local data
            client_h = client_h_encodes[client_id]
            client_hist = history_data[:, :, nodes, :]
            client_future = future_data[:, :, nodes, :] if future_data is not None else None
            
            # Decoder
            pred = client_model.forward_decoder(
                client_hist, client_h, client_graph_enc, client_future, batch_seen
            )
            
            # [Assign Logic] 결과 저장
            pred_all[:, :, nodes, :] = pred
        
        forward_total = fwd_c2s + fwd_s2c
        if train and torch.is_grad_enabled():
            self.total_forward_bytes += forward_total
        
        return {
            'prediction': pred_all,
            'all_h_encode': all_h_flat,
            'graph_encoding': graph_encoding,
            'forward_communication': {
                'client_to_server_bytes': fwd_c2s,
                'server_to_client_bytes': fwd_s2c,
                'total_bytes': forward_total
            }
        }
    
    def compute_backward_comm(self, forward_return: Dict) -> Dict[str, int]:
        """Backward 통신량 (C <- S <- C)"""
        b_c2s = 0
        b_s2c = 0
        
        # C->S: graph_encoding grad
        graph_encoding = forward_return.get('graph_encoding')
        if graph_encoding is not None and graph_encoding.grad is not None:
            b_c2s = graph_encoding.grad.numel() * graph_encoding.grad.element_size()
        
        # S->C: all_h_encode grad
        all_h = forward_return.get('all_h_encode')
        if all_h is not None and all_h.grad is not None:
            b_s2c = all_h.grad.numel() * all_h.grad.element_size()
        
        self.total_backward_bytes += (b_c2s + b_s2c)
        
        return {'client_to_server': b_c2s, 'server_to_client': b_s2c}
    
    def get_communication_stats(self) -> Dict[str, float]:
        return {
            'total_forward_mb': self.total_forward_bytes / (1024 * 1024),
            'total_backward_mb': self.total_backward_bytes / (1024 * 1024),
            'total_mb': (self.total_forward_bytes + self.total_backward_bytes) / (1024 * 1024),
        }
    
    def get_client_models(self) -> List[nn.Module]:
        return list(self.client_models)
    
    def get_server_model(self) -> nn.Module:
        return self.server_model

