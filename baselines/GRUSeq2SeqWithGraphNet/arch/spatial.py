# spatial.py
import torch
import torch.nn as nn
from typing import Optional

from .graph_nets import GraphNet


class GraphNetSpatial(nn.Module):
    """
    GraphNet Spatial: 노드 간 공간적 관계 학습
    
    Input: (B, num_layers, N, hidden_size)
    Output: (B, num_layers, N, hidden_size)
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        gn_layer_num: int = 2,
        gn_hidden_size: int = 256,
        gn_updated_node_size: int = 128,
        gn_updated_edge_size: int = 128,
        gn_updated_global_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
            dropout=dropout,
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        graph: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, num_layers, N, hidden_size)
            graph: Not used (for interface compatibility)
            edge_index: (2, E) edge indices
            edge_attr: (E,) edge attributes
        Returns:
            spatial_features: (B, num_layers, N, hidden_size)
        """
        B, L, N, H = x.shape
        device = x.device
        
        # (B, num_layers, N, hidden_size) -> (N, B, num_layers, hidden_size)
        graph_input = x.permute(2, 0, 1, 3)
        
        # Prepare edge_attr
        if edge_attr is not None:
            edge_attr_expanded = edge_attr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            num_edges = edge_index.shape[1] if edge_index is not None else 0
            edge_attr_expanded = torch.ones(num_edges, 1, 1, 1, device=device)
        
        # GraphNet forward
        graph_encoding = self.gcn(graph_input, edge_index, edge_attr_expanded)
        # (N, B, num_layers, hidden_size)
        
        # (N, B, num_layers, hidden_size) -> (B, num_layers, N, hidden_size)
        output = graph_encoding.permute(1, 2, 0, 3)
        
        return output