# #!/usr/bin/env python3
# """STAEformer 클라이언트 모델들"""

# from typing import Tuple, Optional
# import torch
# import torch.nn as nn

# import sys
# import os

# # sys.path.append(os.path.abspath(__file__ + '/../../..'))
# from ...base_models import BaseClientModel
# from ...layers import SelfAttentionLayer, AttentionPoolLayer, NodeSummaryLayer, TokenExpansionLayer


# class STAEformerClientModel(BaseClientModel):
#     """
#     STAEformer 기본 클라이언트 모델 (Temporal Attention only)
#     """
#     def __init__(
#         self, 
#         num_nodes: int, 
#         in_steps: int = 12, 
#         out_steps: int = 12, 
#         steps_per_day: int = 288,
#         input_dim: int = 3, 
#         output_dim: int = 1,
#         input_embedding_dim: int = 24, 
#         tod_embedding_dim: int = 24, 
#         dow_embedding_dim: int = 24,
#         spatial_embedding_dim: int = 24,
#         adaptive_embedding_dim: int = 0,
#         feed_forward_dim: int = 256, 
#         num_heads: int = 4, 
#         temporal_layers: int = 1, 
#         dropout: float = 0.1
#     ):
#         super().__init__(num_nodes, 0)
        
#         self.num_nodes = num_nodes
#         self.in_steps = in_steps
#         self.out_steps = out_steps
#         self.steps_per_day = steps_per_day
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         self.model_dim = (
#             input_embedding_dim + tod_embedding_dim + dow_embedding_dim +
#             spatial_embedding_dim + adaptive_embedding_dim
#         )
        
#         # Embedding layers
#         self.input_proj = nn.Linear(input_dim, input_embedding_dim)
#         if tod_embedding_dim > 0:
#             self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
#         if dow_embedding_dim > 0:
#             self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
#         if spatial_embedding_dim > 0:
#             self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
#             nn.init.xavier_uniform_(self.node_emb)
#         if adaptive_embedding_dim > 0:
#             self.adaptive_embedding = nn.init.xavier_uniform_(
#                 nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
#             )
        
#         # Temporal Attention layers
#         self.attn_layers_t = nn.ModuleList([
#             SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
#             for _ in range(temporal_layers)
#         ])
    
#     def forward_temporal(self, history_data: torch.Tensor) -> torch.Tensor:
#         """Temporal Attention 수행"""
#         x = history_data
#         batch_size = x.shape[0]
        
#         # Embedding logic
#         if hasattr(self, 'tod_embedding'):
#             tod = x[..., 1] * self.steps_per_day
#         if hasattr(self, 'dow_embedding'):
#             dow = x[..., 2] * 7
#         x = x[..., :self.input_dim]
        
#         x = self.input_proj(x)
#         features = [x]
        
#         if hasattr(self, 'tod_embedding'):
#             features.append(self.tod_embedding(tod.long()))
#         if hasattr(self, 'dow_embedding'):
#             features.append(self.dow_embedding(dow.long()))
#         if hasattr(self, 'node_emb'):
#             spatial_emb = self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
#             features.append(spatial_emb)
#         if hasattr(self, 'adaptive_embedding'):
#             adp_emb = self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape)
#             features.append(adp_emb)
        
#         x = torch.cat(features, dim=-1)
        
#         # Temporal attention (dim=1)
#         for attn in self.attn_layers_t:
#             x = attn(x, dim=1)
            
#         return x # (B, T, N, D)
    
#     def forward_encoder(self, history_data: torch.Tensor) -> torch.Tensor:
#         return self.forward_temporal(history_data)
        
#     def forward_decoder(self, *args, **kwargs) -> torch.Tensor:
#         raise NotImplementedError


# class STAEformerHierClientModel(STAEformerClientModel):
#     """
#     STAEformer-Hier (STAEformer + Pooling) 클라이언트 모델
#     Flow: Temporal -> Spatial -> Pooling -> (Send) -> (Receive) -> Expansion -> Projection
#     """
    
#     def __init__(
#         self, 
#         num_nodes: int, 
#         in_steps: int = 12, 
#         out_steps: int = 12, 
#         steps_per_day: int = 288,
#         input_dim: int = 3, 
#         output_dim: int = 1,
#         input_embedding_dim: int = 24, 
#         tod_embedding_dim: int = 24, 
#         dow_embedding_dim: int = 24,
#         spatial_embedding_dim: int = 0, 
#         adaptive_embedding_dim: int = 80,
#         feed_forward_dim: int = 256, 
#         num_heads: int = 4, 
#         temporal_layers: int = 1, 
#         spatial_layers: int = 1,
#         dropout: float = 0.1,
#         use_mixed_proj: bool = True,
#         num_tokens: int = 4,           # 압축할 토큰 수
#         pooling_method: str = 'attention',
#         subgraph_adj: Optional[torch.Tensor] = None,
#     ):
#         super().__init__(
#             num_nodes, in_steps, out_steps, steps_per_day,
#             input_dim, output_dim,
#             input_embedding_dim, tod_embedding_dim, dow_embedding_dim,
#             spatial_embedding_dim, adaptive_embedding_dim,
#             feed_forward_dim, num_heads, temporal_layers, dropout
#         )
        
#         self.spatial_layers = spatial_layers
#         self.use_mixed_proj = use_mixed_proj
#         self.num_tokens = num_tokens
#         self.pooling_method = pooling_method
        
#         # 1. Local Spatial Attention layers
#         self.attn_layers_s = nn.ModuleList([
#             SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
#             for _ in range(spatial_layers)
#         ])
        
#         # 2. Pooling Layer: (N nodes) -> (K tokens)
#         if pooling_method == 'attention':
#             # AttentionPoolLayer: Learnable Queries로 정보를 요약
#             self.pool_layer = AttentionPoolLayer(
#                 model_dim=self.model_dim,
#                 num_tokens=num_tokens,
#                 num_heads=num_heads,
#                 dropout=dropout
#             )
#             # 3. Expansion Layer: (K tokens) -> (N nodes) 복원
#             # Cross Attention: Query=Local Nodes, Key/Value=Global Tokens
#             self.expansion_layer = TokenExpansionLayer(self.model_dim, num_heads, dropout)
            
#         elif pooling_method == 'simple':
#             if subgraph_adj is None:
#                 subgraph_adj = torch.eye(num_nodes, dtype=torch.float32)
            
#             # buffer로 등록하여 디바이스 이동 자동화
#             self.register_buffer('subgraph_adj', subgraph_adj)
            
#             self.pool_layer = NodeSummaryLayer(
#                 model_dim=self.model_dim,
#                 subgraph_adj=self.subgraph_adj,
#                 num_tokens=num_tokens
#             )
#             self.expansion_layer = nn.Linear(num_tokens, num_nodes)
#         else:
#             raise ValueError(f"Unknown pooling_method: {pooling_method}")
        
#         self.fusion_norm = nn.LayerNorm(self.model_dim)
#         # 4. Output projection
#         if use_mixed_proj:
#             self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
#         else:
#             self.temporal_proj = nn.Linear(in_steps, out_steps)
#             self.output_proj = nn.Linear(self.model_dim, output_dim)
    
#     def forward_temporal_spatial_pool(
#         self, 
#         history_data: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Phase 1: Local Computation & Pooling
        
#         Returns:
#             spatial_features: (B, T, N, D) - 로컬 정보 (보존용)
#             pooled_features: (B, T, K, D) - 서버 전송용 (압축됨)
#         """
#         # 1. Temporal Attention
#         x = self.forward_temporal(history_data) # (B, T, N, D)
        
#         # 2. Local Spatial Attention
#         for attn in self.attn_layers_s:
#             x = attn(x, dim=2)
        
#         # x는 나중에 Residual Connection을 위해 저장해둬야 함 (spatial_features)
        
#         # 3. Pooling (N -> K)
#         if self.pooling_method == 'attention':
#             pooled = self.pool_layer(x) # (B, T, K, D)
#         elif self.pooling_method == 'simple':
#             # (B, T, N, D) -> (B, T, D, N) -> (B, T, D, K) -> (B, T, K, D)
#             pooled = self.pool_layer(x.transpose(2, 3)).transpose(2, 3)
            
#         return x, pooled
    
#     def forward_expansion(self, server_features: torch.Tensor, client_features: torch.Tensor) -> torch.Tensor:
#         """
#         Phase 3: Expansion & Mixing
        
#         Args:
#             server_features: (B, T, K, D) - 서버에서 처리된 Global Context
#             client_features: (B, T, N, D) - 로컬에서 보존한 Local Context
#         Returns:
#             mixed_features: (B, T, N, D)
#         """
#         # 4. Expansion (K -> N) & Mixing
#         if self.pooling_method == 'attention':
#             # Cross Attention: Query=Client(N), Key/Value=Server(K)
#             # Client의 각 노드가 필요한 전역 정보를 Server Token에서 가져옴
#             expanded = self.expansion_layer(client_features, server_features)
#         elif self.pooling_method == 'simple':
#             # Linear Interpolation/Projection
#             expanded = self.expansion_layer(server_features.transpose(2, 3)).transpose(2, 3)
            
#         # Residual Connection: Local Detail + Global Context
#         mixed_features = self.fusion_norm(client_features + expanded)
        
#         return mixed_features

#     def forward_projection(self, mixed_features: torch.Tensor) -> torch.Tensor:
#         """Phase 4: Final Prediction"""
#         x = mixed_features
#         batch_size = x.shape[0]
        
#         if self.use_mixed_proj:
#             out = x.transpose(1, 2)  # (B, N, T, D)
#             out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
#             out = self.output_proj(out).view(
#                 batch_size, self.num_nodes, self.out_steps, -1
#             )
#             out = out.transpose(1, 2) # (B, T_out, N, C_out)
#         else:
#             out = x.transpose(1, 3)
#             out = self.temporal_proj(out)
#             out = self.output_proj(out.transpose(1, 3))
        
#         return out