#!/usr/bin/env python3
"""GRU Seq2Seq + GraphNet Hierarchical Split Learning 서버 모델"""

import torch
import torch.nn as nn

from ...base_models import BaseServerModel
from ...layers import SelfAttentionLayer

class GRUSeq2SeqGraphNetHierServerModel(BaseServerModel):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        feed_forward_dim: int = 256,
    ):
        super().__init__(num_nodes, hidden_dim)
        
        self.attn_layers = nn.ModuleList([
            SelfAttentionLayer(hidden_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        # pooled_features: (B, num_layers, total_tokens_all_clients, hidden_dim)
        x = pooled_features
        
        for attn in self.attn_layers:
            # dim=-2 (token dimension)을 기준으로 Self-Attention 수행
            x = attn(x, dim=-2) 
            
        return x
