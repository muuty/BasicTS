#!/usr/bin/env python3
"""
Generic Split Learning Model

Encoder, Decoder, Server 클래스를 조합하여 Split Learning 구조를 구현합니다.

Graph 처리 전략:
- 서버는 그래프를 생성하거나 보관하지 않습니다.
- Graph가 필요하다면 각 모델의 spatial 모듈에서 직접 생성하거나
  static graph를 초기화 시 주입받아 관리해야 합니다.
"""

import copy
from typing import Dict, List, Optional, Type
from baselines.node_partition import setup_split_learning_nodes
import torch
import torch.nn as nn


class SplitLearningClient(nn.Module):
    """
    Split Learning Client: Encoder + Decoder
    
    구조:
        Forward Phase 1: Encoder(x) → features
        Forward Phase 2: Decoder(processed_features) → prediction
    """
    
    def __init__(
        self,
        encoder_cls: Type[nn.Module],
        decoder_cls: Type[nn.Module],
        encoder_params: Dict,
        decoder_params: Dict,
        num_nodes: int,
    ) -> None:
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Encoder
        enc_params = copy.deepcopy(encoder_params)
        enc_params['num_nodes'] = num_nodes
        self.encoder = encoder_cls(**enc_params)
        
        # Decoder
        dec_params = copy.deepcopy(decoder_params)
        dec_params['num_nodes'] = num_nodes
        self.decoder = decoder_cls(**dec_params)
    
    def forward_encoder(self, history_data: torch.Tensor) -> torch.Tensor:
        """
        Phase 1: Local Encoding
        
        Args:
            history_data: (B, T, N_client, C)
        Returns:
            features: (B, T', N_client, D)
        """
        features, graph = self.encoder(history_data)
        return features
    
    def forward_decoder(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Phase 2: Local Decoding
        
        Args:
            spatial_features: (B, T, N_client, D)
        Returns:
            prediction: (B, T_out, N_client, D_out)
        """
        return self.decoder(spatial_features)


class SplitLearningServer(nn.Module):
    """
    Split Learning Server: Global Spatial Processing
    
    Graph 처리:
    - use_graph=False → Graph 없이 Spatial 처리 (STAEformer)
    - use_graph=True, full_adj_matrix=None → Adaptive graph 생성 (STGformer)
    - use_graph=True, full_adj_matrix 있음 → Static adjacency 사용 (STGCN)
    """
    
    def __init__(
        self,
        spatial_cls: Type[nn.Module],
        spatial_params: Dict,
    ) -> None:
        super().__init__()
        
        self.spatial = spatial_cls(**spatial_params)
    
    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features: (B, T, N_total, D)
        Returns:
            spatial_features: (B, T, N_total, D)
        """
        return self.spatial(global_features, None)


class SplitLearningModel(nn.Module):
    """
    Generic Split Learning Model
    
    구조:
        Client[i]: Encoder → [Server로 전송] → [수신] → Decoder
        Server: Global Spatial Processing (전체 graph 관리)
    """

    def __init__(
        self,
        num_clients: int,
        total_nodes: int,
        encoder_cls: Type[nn.Module],
        decoder_cls: Type[nn.Module],
        spatial_cls: Type[nn.Module],
        encoder_params: Dict,
        decoder_params: Dict,
        server_spatial_params: Dict,
        partition_method: str,
        adj_matrix: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, List[str]]] = None,
        in_steps: int = 12,
        out_steps: int = 12,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        
        self.num_clients = num_clients
        self.total_nodes = total_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim
        
        client_nodes_list, subgraph_adj_list, _ = setup_split_learning_nodes(
            num_nodes=total_nodes,
            num_clients=num_clients,
            grouping_method=partition_method,
            adj_matrix=adj_matrix,
            metadata=metadata,
        )
        self.subgraph_adj_list = subgraph_adj_list

        if len(client_nodes_list) != num_clients:
            print(f"Warning: len(client_nodes_list) != num_clients, {len(client_nodes_list)} != {num_clients}")
            self.num_clients = len(client_nodes_list)
            
        # 노드 인덱스를 버퍼로 등록
        for client_id, client_nodes in enumerate(client_nodes_list):
            self.register_buffer(
                f'client_nodes_{client_id}',
                torch.tensor(client_nodes, dtype=torch.long)
            )
        
        # Client 모델들 생성
        self.client_models = nn.ModuleList()
        for client_id, client_nodes in enumerate(client_nodes_list):
            # STGCN 같은 graph 기반 encoder는 local num_nodes에 맞는 subgraph adjacency가 필요함.
            # (전체 207x207 adj를 그대로 주면 local N(예:21)과 shape mismatch로 einsum이 터짐)
            encoder_params_for_client = encoder_params
            if subgraph_adj_list is not None and client_id < len(subgraph_adj_list):
                encoder_params_for_client = dict(encoder_params)
                encoder_params_for_client["adj_matrix"] = subgraph_adj_list[client_id]

            client_model = SplitLearningClient(
                encoder_cls=encoder_cls,
                decoder_cls=decoder_cls,
                encoder_params=encoder_params_for_client,
                decoder_params=decoder_params,
                num_nodes=len(client_nodes),
            )
            self.client_models.append(client_model)
        
        # Server 모델 생성
        self.server_model = SplitLearningServer(
            spatial_cls=spatial_cls,
            spatial_params=server_spatial_params,
        )
        
        # 통신량 추적
        self.total_forward_bytes = 0
        self.total_backward_bytes = 0
        
        self._print_init_info()

    def _print_init_info(self):
        """초기화 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        client_params = sum(
            sum(p.numel() for p in client.parameters()) 
            for client in self.client_models
        )
        server_params = sum(p.numel() for p in self.server_model.parameters())
        
        print(f"\n🔗 SplitLearningModel 초기화")
        print(f"  - 클라이언트 수: {self.num_clients}")
        print(f"  - 총 파라미터: {total_params:,}")
        print(f"    - Client 파라미터: {client_params:,}")
        print(f"    - Server 파라미터: {server_params:,}")

    def get_client_nodes(self, client_id: int) -> torch.Tensor:
        """클라이언트별 노드 인덱스 반환"""
        return getattr(self, f'client_nodes_{client_id}')

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor] = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs,
    ) -> Dict:
        B, T_in, N, C = history_data.shape
        device = history_data.device
        dtype = history_data.dtype

        # ─────────────────────────────────────────────────────────────
        # Phase 1: Client Encoding
        # ─────────────────────────────────────────────────────────────
        client_features: Dict[int, torch.Tensor] = {}
        
        for client_id in range(self.num_clients):
            nodes = self.get_client_nodes(client_id)
            x_client = history_data[:, :, nodes, :]
            
            feat = self.client_models[client_id].forward_encoder(x_client)
            client_features[client_id] = feat

        # ─────────────────────────────────────────────────────────────
        # Phase 2: Aggregate to Global Features (원래 노드 위치 유지)
        # ─────────────────────────────────────────────────────────────
        sample_feat = client_features[0]
        T_feat = sample_feat.shape[1]
        D_feat = sample_feat.shape[-1]
        
        global_features = torch.zeros(
            B, T_feat, self.total_nodes, D_feat,
            device=device, dtype=dtype
        )
        
        for client_id in range(self.num_clients):
            nodes = self.get_client_nodes(client_id)
            global_features[:, :, nodes, :] = client_features[client_id]
        
        # 통신량 (Client → Server)
        fwd_c2s = global_features.numel() * global_features.element_size()
        
        if train and torch.is_grad_enabled() and global_features.requires_grad:
            global_features.retain_grad()

        # ─────────────────────────────────────────────────────────────
        # Phase 3: Server Processing
        # ─────────────────────────────────────────────────────────────
        spatial_features = self.server_model(global_features)
        
        # 통신량 (Server → Client)
        fwd_s2c = spatial_features.numel() * spatial_features.element_size()
        
        if train and torch.is_grad_enabled() and spatial_features.requires_grad:
            spatial_features.retain_grad()

        # ─────────────────────────────────────────────────────────────
        # Phase 4: Client Decoding
        # ─────────────────────────────────────────────────────────────
        prediction = torch.zeros(
            B, self.out_steps, self.total_nodes, self.output_dim,
            device=device, dtype=dtype
        )
        
        for client_id in range(self.num_clients):
            nodes = self.get_client_nodes(client_id)
            client_spatial = spatial_features[:, :, nodes, :]
            
            pred = self.client_models[client_id].forward_decoder(client_spatial)
            prediction[:, :, nodes, :] = pred

        # ─────────────────────────────────────────────────────────────
        # 통신량 누적
        # ─────────────────────────────────────────────────────────────
        forward_total = fwd_c2s + fwd_s2c
        if train and torch.is_grad_enabled():
            self.total_forward_bytes += forward_total

        return {
            'prediction': prediction,
            'global_features': global_features,
            'spatial_features': spatial_features,
            'forward_communication': {
                'client_to_server_bytes': fwd_c2s,
                'server_to_client_bytes': fwd_s2c,
                'total_bytes': forward_total,
            },
        }

    def compute_backward_comm(self, forward_return: Dict) -> Dict[str, int]:
        """Backward 통신량 계산"""
        b_c2s = 0
        b_s2c = 0
        
        spatial_features = forward_return.get('spatial_features')
        if spatial_features is not None and spatial_features.grad is not None:
            b_c2s = spatial_features.grad.numel() * spatial_features.grad.element_size()
        
        global_features = forward_return.get('global_features')
        if global_features is not None and global_features.grad is not None:
            b_s2c = global_features.grad.numel() * global_features.grad.element_size()
        
        self.total_backward_bytes += (b_c2s + b_s2c)
        
        return {
            'client_to_server': b_c2s,
            'server_to_client': b_s2c,
            'total': b_c2s + b_s2c,
        }

    def get_communication_stats(self) -> Dict[str, float]:
        """총 통신량 통계 반환 (MB 단위)"""
        return {
            'total_forward_mb': self.total_forward_bytes / (1024 * 1024),
            'total_backward_mb': self.total_backward_bytes / (1024 * 1024),
            'total_mb': (self.total_forward_bytes + self.total_backward_bytes) / (1024 * 1024),
        }
    
    def reset_communication_stats(self):
        """통신량 통계 초기화"""
        self.total_forward_bytes = 0
        self.total_backward_bytes = 0

    def get_client_models(self) -> List[nn.Module]:
        """클라이언트 모델 리스트 반환 (Runner 호환)"""
        return list(self.client_models)

    def get_server_model(self) -> nn.Module:
        """서버 모델 반환 (Runner 호환)"""
        return self.server_model
