#!/usr/bin/env python3
"""
Generic HierSplit Model

Encoder, Spatial, Decoder 클래스를 조합하여 HierSplit 구조를 구현합니다.
Server는 Client와 동일한 Spatial 클래스를 재사용합니다.
"""

import copy
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from baselines.layers import AttentionPoolLayer, NodeSummaryLayer, TokenExpansionLayer
from baselines.node_partition import setup_split_learning_nodes


class HierSplitServer(nn.Module):
    """
    Server: Transformer self-attention on pooled tokens
    backbone과 무관하게 고정된 구조
    """
    def __init__(self, model_dim: int, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__()
        # 논문 Eq. 14: Transformer encoder layer
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(model_dim)
    
    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_features: (B, T, Total_K, D)
        """
        B, T, K, D = pooled_features.shape
        
        # (B, T, K, D) → (B*T, K, D) for attention
        x = pooled_features.reshape(B * T, K, D)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        
        return x.reshape(B, T, K, D)


class HierSplitClient(nn.Module):
    """
    HierSplit Client: Encoder → Spatial → Pooling → [Server] → Expansion → Decoder
    """

    def __init__(
        self,
        encoder_cls: Type[nn.Module],
        spatial_cls: Type[nn.Module],
        decoder_cls: Type[nn.Module],
        encoder_params: Dict,
        spatial_params: Dict,
        decoder_params: Dict,
        model_dim: int,
        num_nodes: int,
        num_tokens: int,
        pooling_method: str,
        num_heads: int,
        num_clients: int,
        dropout: float,
        subgraph_adj: Optional[torch.Tensor] = None,
        fuse_ln: bool = True,
    ) -> None:
        super().__init__()

        # Core modules
        print("encoder params")
        print(encoder_params)
        self.encoder = encoder_cls(**encoder_params)
        self.spatial = spatial_cls(**spatial_params)
        self.decoder = decoder_cls(**decoder_params)
        
        # Config
        self.model_dim = model_dim
        self.num_nodes = num_nodes
        if num_tokens == "auto":
            self.num_tokens = max(1, int(round(num_nodes / num_clients / 5)))
        else:
            self.num_tokens = num_tokens
        self.pooling_method = pooling_method

        # Pooling & Expansion layers
        if pooling_method == "attention":
            self.pool_layer = AttentionPoolLayer(
                model_dim=model_dim,
                num_tokens=self.num_tokens,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.expansion_layer = TokenExpansionLayer(
                model_dim=model_dim, 
                num_heads=num_heads, 
                dropout=dropout,
            )
        elif pooling_method == "linear":
            if subgraph_adj is None:
                subgraph_adj = torch.eye(num_nodes)
            self.register_buffer("subgraph_adj", subgraph_adj)
            self.pool_layer = NodeSummaryLayer(
                model_dim=model_dim,
                subgraph_adj=self.subgraph_adj,
                num_tokens=self.num_tokens,
            )
            self.expansion_layer = nn.Linear(self.num_tokens, num_nodes)
        else:
            raise ValueError(f"Unknown pooling_method: {pooling_method}")
        
        # Fusion
        if fuse_ln:
            self.fusion_norm = nn.LayerNorm(model_dim)
        else:
            self.fusion_norm = nn.Identity()

    def forward_temporal_spatial_pool(
        self, history_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1: Local Processing & Pooling
        
        Args:
            history_data: (B, T, N, C)
        Returns:
            spatial_features: (B, T, N, D) - saved for expansion
            pooled_features: (B, T, K, D) - sent to server
        """
        # Encoder: Embedding + Temporal Attention
        encoded, graph = self.encoder(history_data)
        
        # Local Spatial Attention
        spatial_out = self.spatial(encoded, graph)
        
        # Pooling (N → K)
        pooled = self.pool_layer(spatial_out)
        
        return spatial_out, pooled

    def forward_expansion(
        self, 
        server_tokens: torch.Tensor, 
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase 3: Expansion & Mixing
        
        Args:
            server_tokens: (B, T, K, D) - from server
            local_features: (B, T, N, D) - saved from phase 1
        Returns:
            mixed_features: (B, T, N, D)
        """
        # Expansion (K → N)
        if self.pooling_method == "attention":
            # Cross attention: Query=local, Key/Value=server
            expanded = self.expansion_layer(local_features, server_tokens)
        else:
            # Linear: (B, T, K, D) → (B, T, D, K) → (B, T, D, N) → (B, T, N, D)
            expanded = self.expansion_layer(server_tokens.transpose(2, 3)).transpose(2, 3)
        
        # Residual connection
        mixed = self.fusion_norm(local_features + expanded)
        return mixed

    def forward_projection(self, features: torch.Tensor) -> torch.Tensor:
        """
        Phase 4: Final Prediction
        
        Args:
            features: (B, T, N, D)
        Returns:
            prediction: (B, T_out, N, output_dim)
        """
        return self.decoder(features)


class HierSplitModel(nn.Module):
    """
    Generic HierSplit Model
    
    구조:
        Client: Encoder → Local Spatial → Pooling → [전송]
        Server: Global Spatial on Tokens
        Client: ← [수신] → Expansion → Decoder
    
    사용 예시:
        model = HierSplitModel(
            num_clients=10,
            client_nodes_list=[[0,1,2,...], [20,21,...], ...],
            total_nodes=207,
            encoder_cls=STAEformerEncoder,
            spatial_cls=STAEformerSpatial,
            decoder_cls=STAEformerDecoder,
            encoder_params={...},
            spatial_params={...},
            decoder_params={...},
            server_spatial_params={...},  # Server용 Spatial 파라미터
            model_dim=96,
            num_tokens=4,
            ...
        )
    """

    def __init__(
        self,
        # Client 구성
        num_clients: int,
        total_nodes: int,
        encoder_cls: Type[nn.Module],
        spatial_cls: Type[nn.Module],
        decoder_cls: Type[nn.Module],
        encoder_params: Dict,
        spatial_params: Dict,
        decoder_params: Dict,
        # Server 구성 (Spatial 재사용)
        server_spatial_params: Dict,
        # Pooling 설정
        model_dim: int,
        num_tokens: int,
        partition_method: str,
        pooling_method: str = "attention",
        num_heads: int = 4,
        dropout: float = 0.1,
        adj_matrix: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, List[str]]] = None,
        # Output 설정
        out_steps: int = 12,
        output_dim: int = 1,
        fuse_ln: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_clients = num_clients
        self.total_nodes = total_nodes
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.model_dim = model_dim

        client_nodes_list, subgraph_adj_list, _ = setup_split_learning_nodes(
            num_nodes=self.total_nodes,
            num_clients=self.num_clients,
            grouping_method=partition_method,
            adj_matrix=adj_matrix,
            metadata=metadata,
        )

        if self.num_clients != len(client_nodes_list):
            print(f"Warning: num_clients != len(client_nodes_list), {self.num_clients} != {len(client_nodes_list)}")
            self.num_clients = len(client_nodes_list)

        # 각 클라이언트별 토큰 수 계산 (auto인 경우 클라이언트마다 다를 수 있음)
        self.client_token_counts: List[int] = []
        for client_nodes in client_nodes_list:
            num_client_nodes = len(client_nodes)
            if num_tokens == "auto":
                client_num_tokens = max(1, int(round(num_client_nodes / num_clients / 5)))
            else:
                client_num_tokens = num_tokens
            self.client_token_counts.append(client_num_tokens)

        # 노드 인덱스를 버퍼로 등록 (GPU 이동 시 함께 이동)
        for client_id, client_nodes in enumerate(client_nodes_list):
            self.register_buffer(
                f'client_nodes_{client_id}',
                torch.tensor(client_nodes, dtype=torch.long)
            )

        # ─────────────────────────────────────────────────────────────
        # Client 모델들 생성
        # ─────────────────────────────────────────────────────────────
        self.client_models = nn.ModuleList()
        
        for client_id, client_nodes in enumerate(client_nodes_list):
            # 파라미터 복사
            enc_p = copy.deepcopy(encoder_params)
            spa_p = copy.deepcopy(spatial_params)
            dec_p = copy.deepcopy(decoder_params)

            # num_nodes 설정
            num_client_nodes = len(client_nodes)
            enc_p["num_nodes"] = num_client_nodes
            dec_p["num_nodes"] = num_client_nodes

            # Subgraph adjacency
            sub_adj = subgraph_adj_list[client_id]
            enc_p["adj_matrix"] = sub_adj


            client_model = HierSplitClient(
                encoder_cls=encoder_cls,
                spatial_cls=spatial_cls,
                decoder_cls=decoder_cls,
                encoder_params=enc_p,
                spatial_params=spa_p,
                decoder_params=dec_p,
                model_dim=model_dim,
                num_nodes=num_client_nodes,
                num_tokens=num_tokens,
                pooling_method=pooling_method,
                num_heads=num_heads,
                dropout=dropout,
                subgraph_adj=sub_adj,
                fuse_ln=fuse_ln,
                num_clients=num_clients,
            )
            self.client_models.append(client_model)

        # ─────────────────────────────────────────────────────────────
        # Server 모델 생성 (Spatial 클래스 재사용)
        # ─────────────────────────────────────────────────────────────
        self.server_model = HierSplitServer(
            **server_spatial_params,
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
        
        print(f"\n🔗 HierSplitModel 초기화")
        print(f"  - 클라이언트 수: {self.num_clients}")
        print(f"  - 전체 노드 수: {self.total_nodes}")
        print(f"  - 토큰 수 (per client): {self.client_token_counts}")
        print(f"  - 총 토큰 수: {sum(self.client_token_counts)}")
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
        """
        Forward pass
        
        Args:
            history_data: (B, T_in, N, C)
            future_data: Not used
            batch_seen: Current batch index
            epoch: Current epoch
            train: Training mode flag
            
        Returns:
            Dict containing prediction and communication info
        """
        B, T_in, N, C = history_data.shape
        device = history_data.device
        dtype = history_data.dtype

        # ─────────────────────────────────────────────────────────────
        # Phase 1: Client Local Processing & Pooling
        # ─────────────────────────────────────────────────────────────
        client_spatial_contexts: Dict[int, torch.Tensor] = {}
        client_pooled_outputs: List[torch.Tensor] = []

        for client_id in range(self.num_clients):
            client_model = self.client_models[client_id]
            nodes = self.get_client_nodes(client_id)
            client_history = history_data[:, :, nodes, :]
            
            spatial_feat, pooled_feat = client_model.forward_temporal_spatial_pool(
                client_history
            )
            client_spatial_contexts[client_id] = spatial_feat
            client_pooled_outputs.append(pooled_feat)

        # ─────────────────────────────────────────────────────────────
        # Phase 2: Aggregate & Send to Server
        # ─────────────────────────────────────────────────────────────
        # (B, T, K, D) × num_clients → (B, T, Total_K, D)
        all_pooled = torch.cat(client_pooled_outputs, dim=2)
        
        # 통신량 계산 (Client → Server)
        fwd_c2s = all_pooled.numel() * all_pooled.element_size()
        
        # Backward를 위해 gradient 유지
        if train and torch.is_grad_enabled() and all_pooled.requires_grad:
            all_pooled.retain_grad()

        # ─────────────────────────────────────────────────────────────
        # Phase 3: Server Global Processing
        # ─────────────────────────────────────────────────────────────
        server_features = self.server_model(all_pooled)
        
        # 통신량 계산 (Server → Client)
        fwd_s2c = server_features.numel() * server_features.element_size()
        
        # Backward를 위해 gradient 유지
        if train and torch.is_grad_enabled() and server_features.requires_grad:
            server_features.retain_grad()

        # ─────────────────────────────────────────────────────────────
        # Phase 4: Distribute & Client Expansion/Projection
        # ─────────────────────────────────────────────────────────────
        # Server features를 클라이언트별로 분할
        server_features_split = torch.split(server_features, self.client_token_counts, dim=2)
        
        # 최종 prediction 텐서
        pred_all = torch.zeros(
            B, self.out_steps, self.total_nodes, self.output_dim,
            device=device, dtype=dtype
        )

        for client_id in range(self.num_clients):
            client_model = self.client_models[client_id]
            nodes = self.get_client_nodes(client_id)
            
            # 해당 클라이언트의 server features
            my_server_feat = server_features_split[client_id]
            # 저장해둔 local features
            my_local_feat = client_spatial_contexts[client_id]
            
            # Expansion & Mixing
            mixed = client_model.forward_expansion(my_server_feat, my_local_feat)
            
            # Final Projection
            pred = client_model.forward_projection(mixed)
            
            # 결과 저장
            pred_all[:, :, nodes, :] = pred

        # ─────────────────────────────────────────────────────────────
        # Communication Tracking
        # ─────────────────────────────────────────────────────────────
        forward_total = fwd_c2s + fwd_s2c
        if train and torch.is_grad_enabled():
            self.total_forward_bytes += forward_total

        return {
            "prediction": pred_all,
            "all_pooled_features": all_pooled,
            "server_features": server_features,
            "forward_communication": {
                "client_to_server_bytes": fwd_c2s,
                "server_to_client_bytes": fwd_s2c,
                "total_bytes": forward_total,
            },
        }

    def compute_backward_comm(self, forward_return: Dict) -> Dict[str, int]:
        """
        Backward 통신량 계산
        
        Flow:
        1. Client Decoder backprop → server_features.grad
        2. [C→S] server_features.grad 전송
        3. Server backprop → all_pooled.grad
        4. [S→C] all_pooled.grad 전송
        5. Client Encoder backprop
        """
        b_c2s = 0
        b_s2c = 0
        
        # Client → Server: server_features의 gradient
        server_features = forward_return.get("server_features")
        if server_features is not None and server_features.grad is not None:
            b_c2s = server_features.grad.numel() * server_features.grad.element_size()
        
        # Server → Client: all_pooled의 gradient
        all_pooled = forward_return.get("all_pooled_features")
        if all_pooled is not None and all_pooled.grad is not None:
            b_s2c = all_pooled.grad.numel() * all_pooled.grad.element_size()
        
        self.total_backward_bytes += (b_c2s + b_s2c)
        
        return {
            "client_to_server": b_c2s, 
            "server_to_client": b_s2c,
            "total": b_c2s + b_s2c,
        }

    def get_communication_stats(self) -> Dict[str, float]:
        """총 통신량 통계 반환 (MB 단위)"""
        return {
            "total_forward_mb": self.total_forward_bytes / (1024 * 1024),
            "total_backward_mb": self.total_backward_bytes / (1024 * 1024),
            "total_mb": (self.total_forward_bytes + self.total_backward_bytes) / (1024 * 1024),
        }
    
    def reset_communication_stats(self):
        """통신량 통계 초기화"""
        self.total_forward_bytes = 0
        self.total_backward_bytes = 0

    def get_client_models(self) -> List[nn.Module]:
        """클라이언트 모델 리스트 반환"""
        return list(self.client_models)

    def get_server_model(self) -> nn.Module:
        """서버 모델 반환"""
        return self.server_model