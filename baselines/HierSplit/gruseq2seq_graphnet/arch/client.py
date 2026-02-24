#!/usr/bin/env python3
"""GRU Seq2Seq + GraphNet Hierarchical Split Learning 클라이언트 모델"""

import numpy as np
import torch
import torch.nn as nn

from ...base_models import BaseClientModel
from ...layers import AttentionPoolLayer, NodeSummaryLayer, TokenExpansionLayer
from .layers import GraphNet

class GRUSeq2SeqGraphNetHierClientModel(BaseClientModel):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int = 1,
        hidden_size: int = 128,
        gru_num_layers: int = 2,
        dropout: float = 0.0,
        cl_decay_steps: int = 1000,
        use_curriculum_learning: bool = True,
        # --- GraphNet 관련 파라미터 (Base Model과 동일하게 맞춤) ---
        gn_layer_num: int = 2,
        gn_hidden_size: int = 256,
        gn_updated_node_size: int = 128,
        gn_updated_edge_size: int = 128,
        gn_updated_global_size: int = 128,
        # -------------------------------------------------------
        seq_len: int = 12,
        horizon: int = 12,
        num_tokens: int = 1,
        pooling_method: str = 'attention',
        subgraph_adj: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ):
        super().__init__(num_nodes, hidden_size)
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.gru_num_layers = gru_num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.num_tokens = num_tokens
        self.pooling_method = pooling_method
        
        # Buffer 등록 (GPU 이동 자동화)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        self.register_buffer('subgraph_adj', subgraph_adj)

        # 1. GRU Encoder
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=False
        )
        
        # 2. GraphNet (Base Model의 초기화 방식 그대로 적용)
        self.gcn = GraphNet(
            node_input_size=hidden_size,
            edge_input_size=1,
            global_input_size=hidden_size,
            hidden_size=gn_hidden_size,
            updated_node_size=gn_updated_node_size,
            updated_edge_size=gn_updated_edge_size,
            updated_global_size=gn_updated_global_size,
            node_output_size=hidden_size,
            gn_layer_num=gn_layer_num,
            activation='ReLU',
            dropout=dropout
        )
        
        # 3. Pooling & Expansion Layer (Hierarchical Split 추가 부분)
        if pooling_method == 'attention':
            self.pool_layer = AttentionPoolLayer(
                model_dim=hidden_size,
                num_tokens=num_tokens,
                num_heads=4,
                dropout=dropout
            )
            # Server Feature(Token) -> Node Feature 확장용
            self.expansion_layer = TokenExpansionLayer(hidden_size, 4, dropout)
            
        elif pooling_method == 'simple':
            #if self.subgraph_adj is None:
            self.subgraph_adj = torch.eye(num_nodes, dtype=torch.float32)
                
            self.pool_layer = NodeSummaryLayer(
                model_dim=hidden_size,
                subgraph_adj=self.subgraph_adj,
                num_tokens=num_tokens
            )
            self.expansion_layer = nn.Linear(num_tokens, num_nodes)
        else:
            raise ValueError(f"Unknown pooling_method: {pooling_method}")
        
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
        # 4. GRU Decoder (Base Model과 동일: hidden_size * 2)
        self.decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=2 * hidden_size, 
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output Projection
        self.out_net = nn.Linear(2 * hidden_size, output_dim)
    
    def _compute_sampling_threshold(self, batches_seen):
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward_encoder(self, history_data: torch.Tensor):
        """Encoder Only (Used internally)"""
        B, L, N, C = history_data.shape
        # [B, L, N, C] -> [L, B, N, C] -> [L, B*N, C]
        x_input = history_data.permute(1, 0, 2, 3).reshape(L, B * N, C)
        _, h_encode = self.encoder(x_input)
        return h_encode
    
    def forward_encoder_graphnet_pool(
        self, 
        history_data: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ):
        """
        Phase 1: Encoder -> GraphNet -> Pooling
        """
        B, L, N, C = history_data.shape
        
        # Topology 설정 (인자가 없으면 저장된 버퍼 사용)
        curr_edge_index = edge_index if edge_index is not None else self.edge_index
        curr_edge_attr = edge_attr if edge_attr is not None else self.edge_attr
        
        # 1. GRU Encoding
        h_encode = self.forward_encoder(history_data)  # [num_layers, B*N, H]
        
        # 2. GraphNet Pre-processing (Base Model 로직 준수)
        # Reshape: [num_layers, B*N, H] -> [N, B, num_layers, H]
        # (GraphNet은 [Node, Batch, Layer, Feature] 입력을 받음)
        graph_input = h_encode.view(self.gru_num_layers, B, N, self.hidden_size)
        graph_input = graph_input.permute(2, 1, 0, 3) # [N, B, L, F]
        
        # Edge Attr Expansion (Base Model 로직 준수)
        if curr_edge_attr is not None:
             # [E] or [E, F] -> [E, 1, 1, 1] or [E, 1, 1, F]
             # Base Model은 [E] -> [E, 1, 1, 1]로 확장한다고 가정
            if curr_edge_attr.dim() == 1:
                edge_attr_expanded = curr_edge_attr.view(-1, 1, 1, 1)
            else:
                edge_attr_expanded = curr_edge_attr.unsqueeze(1).unsqueeze(1)
        else:
            # Create dummy edge attributes if not provided
            num_edges = curr_edge_index.shape[1] if curr_edge_index is not None else 0
            edge_attr_expanded = torch.ones(num_edges, 1, 1, 1, device=history_data.device)
            
        # 3. GraphNet Forward
        # Output: [N, B, L, F]
        graph_encoding_raw = self.gcn(graph_input, curr_edge_index, edge_attr_expanded)
        
        # 4. Post-processing for Pooling
        # Pooling은 보통 [B, L, N, F] 형태를 선호하므로 이에 맞게 변환
        # [N, B, L, F] -> [B, L, N, F]
        graph_encoding_for_pool = graph_encoding_raw.permute(1, 2, 0, 3)
        
        # 5. Pooling (num_tokens=0일 때는 건너뛰기 - Independent Learning 모드)
        if self.num_tokens == 0:
            # Independent Learning: pooling 없이 빈 텐서 반환
            # shape: [B, L, 0, H] (dim=2가 0인 빈 텐서)
            pooled_features = torch.empty(B, L, 0, self.hidden_size, device=history_data.device)
        else:
            # [B, L, N, F] -> [B, L, Tokens, F]
            pooled_features = self.pool_layer(graph_encoding_for_pool)
        
        # 반환:
        # h_encode: 원본 GRU Hidden [layers, B*N, H]
        # graph_encoding_for_pool: GCN 결과 [B, L, N, H] (나중에 Decoder에서 사용)
        # pooled_features: 서버로 보낼 데이터 [B, L, Tokens, H] (num_tokens=0이면 빈 텐서)
        return h_encode, graph_encoding_for_pool, pooled_features

    def forward_decoder(
        self,
        history_data: torch.Tensor,
        h_encode: torch.Tensor,
        graph_encoding: torch.Tensor, # [B, L, N, H]
        server_features: torch.Tensor, # [B, L, Tokens, H]
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
    ) -> torch.Tensor:
        """
        Phase 2: Expansion -> Fusion -> Decoder
        """
        B, L, N, C = history_data.shape
        
        # 1. Feature Expansion & Fusion
        # num_tokens=0일 때는 Independent Learning 모드: Server Feature 없이 GraphNet 결과만 사용
        if self.num_tokens == 0 or (server_features is not None and server_features.shape[2] == 0):
            # Independent Learning: Server 없이 GraphNet 결과만 사용
            #fused_encoding = self.fusion_norm(graph_encoding)
            fused_encoding = graph_encoding
        else:
            # Server Feature를 Node 레벨로 확장하여 GraphNet 결과와 결합
            if self.pooling_method == 'attention':
                # Query: GraphNet Output [B, L, N, H]
                # Key/Val: Server Features [B, L, Tokens, H]
                server_expanded = self.expansion_layer(
                    graph_encoding, server_features
                ) # [B, L, N, H]
            elif self.pooling_method == 'simple':
                # Linear Expansion
                # [B, L, Tokens, H] -> [B, L, N, H]
                server_expanded = self.expansion_layer(
                    server_features.transpose(2, 3)
                ).transpose(2, 3)
            else:
                raise ValueError(f"Unknown pooling_method")
            
            # Residual Connection: GraphNet Output + Server Feedback
            fused_encoding = graph_encoding + server_expanded
            #fused_encoding = self.fusion_norm(graph_encoding + server_expanded)
        
        # 2. Reshape for Decoder Fusion (Base Model Logic)
        # [B, L, N, H] -> [L, B, N, H] -> [L, B*N, H]
        # Base Model: permute(2, 1, 0, 3) from [N, B, L, F] -> [L, B, N, F] logic equivalent
        fused_encoding = fused_encoding.permute(1, 0, 2, 3).reshape(
            self.gru_num_layers, B * N, self.hidden_size
        )
        
        # 3. Concatenate (Base Model Logic)
        # [L, B*N, 2*H]
        h_decode = torch.cat([h_encode, fused_encoding], dim=-1)
        
        # 4. Decoder Loop (Base Model Logic 그대로 적용)
        x_input = history_data.permute(1, 0, 2, 3).reshape(L, B * N, C)
        
        if self.training and not self.use_curriculum_learning and future_data is not None:
             # Teacher forcing without CL
            y_input = future_data[..., :self.input_dim]
            y_input = y_input.permute(1, 0, 2, 3).reshape(self.horizon, B * N, -1)
            decoder_input = torch.cat([x_input[-1:], y_input[:-1]], dim=0)
            
            out_hidden, _ = self.decoder(decoder_input, h_decode)
            out = self.out_net(out_hidden)
            out = out.view(self.horizon, B, N, self.output_dim)
            out = out.permute(1, 0, 2, 3)
        else:
            # Autoregressive / Curriculum Learning
            last_input = x_input[-1:]
            last_hidden = h_decode
            out_steps = []
            
            if future_data is not None:
                y_gt = future_data[..., :self.input_dim]
                y_gt = y_gt.permute(1, 0, 2, 3).reshape(self.horizon, B * N, -1)
            
            for t in range(self.horizon):
                out_hidden, last_hidden = self.decoder(last_input, last_hidden)
                out = self.out_net(out_hidden)
                out_steps.append(out)
                
                # Next Input Prep
                if self.training and self.use_curriculum_learning and future_data is not None:
                    p_gt = self._compute_sampling_threshold(batch_seen)
                    if np.random.uniform(0, 1) < p_gt:
                        last_input = y_gt[t:t+1]
                    else:
                        last_input = self._prepare_next_input(out, B, N)
                else:
                    last_input = self._prepare_next_input(out, B, N)
            
            out = torch.cat(out_steps, dim=0)
            out = out.view(self.horizon, B, N, self.output_dim).permute(1, 0, 2, 3)
            
        return out

    def _prepare_next_input(self, out, B, N):
        # 차원 패딩 유틸리티 (Base Model 로직 보존)
        if out.shape[-1] < self.input_dim:
            padding = torch.zeros(1, B * N, self.input_dim - out.shape[-1], device=out.device)
            return torch.cat([out, padding], dim=-1)
        else:
            return out[..., :self.input_dim]