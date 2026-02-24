#!/usr/bin/env python3
"""GraphNet Split Learning 서버 모델"""

import torch
import torch.nn as nn

from ...base_models import BaseServerModel
from .layers import GraphNet


class GraphNetSplitServerModel(BaseServerModel):
    """
    GraphNet Split Learning 서버 모델
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 128,
        # GraphNet specific params
        gn_layer_num: int = 2,
        gn_hidden_size: int = 256,
        gn_updated_node_size: int = 128,
        gn_updated_edge_size: int = 128,
        gn_updated_global_size: int = 128,
        dropout: float = 0.0,
        # Graph Structure
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ):
        super().__init__(num_nodes, hidden_dim)
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 저장해두고 forward에서 사용
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            self.edge_index = None
            
        if edge_attr is not None:
            self.register_buffer('edge_attr', edge_attr)
        else:
            self.edge_attr = None
        
        # GraphNet 초기화 (사용자 정의 GraphNet 클래스 시그니처에 맞춤)
        # Node Input/Output size는 GRU hidden size와 동일하다고 가정
        self.graph_net = GraphNet(
            node_input_size=hidden_dim,
            edge_input_size=1,  # 보통 weight 1개
            global_input_size=hidden_dim,
            hidden_size=gn_hidden_size,
            updated_node_size=gn_updated_node_size,
            updated_edge_size=gn_updated_edge_size,
            updated_global_size=gn_updated_global_size,
            node_output_size=hidden_dim,
            gn_layer_num=gn_layer_num,
            activation='ReLU',
            dropout=dropout
        )
    
    def forward(
        self,
        h_encode: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h_encode: (num_layers, B*N, hidden_dim) - Flattened
            edge_index: (2, E)
            edge_attr: (E, ...)
        
        Returns:
            graph_encoding: (num_layers, B*N, hidden_dim)
        """
        # Default graph structure usage
        if edge_index is None:
            edge_index = self.edge_index
        if edge_attr is None:
            edge_attr = self.edge_attr
            
        num_layers, BN, H = h_encode.shape
        B = BN // self.num_nodes
        N = self.num_nodes
        
        # 1. Reshape for GraphNet Input
        # (Layers, B*N, H) -> (Layers, B, N, H) -> (N, B, Layers, H)
        # GraphNet.forward 문서: Node features [N, B, L, F]
        graph_input = h_encode.view(num_layers, B, N, H).permute(2, 1, 0, 3)
        
        # 2. Prepare edge_attr (Expand dims needed for MetaLayer)
        # Expected: [E, 1, 1, 1]
        if edge_attr is not None:
            # 만약 (E) 형태라면 (E, 1, 1, 1)로 확장
            if edge_attr.dim() == 1:
                edge_attr_expanded = edge_attr.view(-1, 1, 1, 1)
            else:
                edge_attr_expanded = edge_attr
        else:
            # Dummy attributes
            num_edges = edge_index.shape[1] if edge_index is not None else 0
            edge_attr_expanded = torch.ones(num_edges, 1, 1, 1, device=h_encode.device)
            
        # 3. GraphNet Forward
        # Output: [N, B, L, F] (updated features)
        graph_out = self.graph_net(graph_input, edge_index, edge_attr_expanded)
        
        # 4. Reshape back
        # [N, B, L, H] -> [L, B, N, H] -> [L, B*N, H]
        graph_encoding = graph_out.permute(2, 1, 0, 3).reshape(num_layers, B * N, H)
        
        return graph_encoding
