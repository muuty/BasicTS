# encoder.py
import torch
import torch.nn as nn
from typing import Tuple, Optional


class GRUEncoder(nn.Module):
    """
    GRU Encoder: 시계열 데이터를 hidden state로 인코딩
    
    Input: (B, T, N, C)
    Output: (features, None)
        - features: (B, num_layers, N, hidden_size) - GRU hidden states
        - graph: None (이 모델은 외부 graph 사용)
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
        )
    
    def forward(self, history_data: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            history_data: (B, T, N, C)
        Returns:
            hidden_states: (B, num_layers, N, hidden_size)
            graph: None
        """
        B, T, N, C = history_data.shape
        
        # (B, T, N, C) -> (T, B*N, C)
        x = history_data.permute(1, 0, 2, 3).reshape(T, B * N, C)
        
        # GRU encoding
        _, h_encode = self.gru(x)  # h_encode: (num_layers, B*N, hidden_size)
        
        # (num_layers, B*N, hidden_size) -> (B, num_layers, N, hidden_size)
        h_encode = h_encode.view(self.num_layers, B, N, self.hidden_size)
        h_encode = h_encode.permute(1, 0, 2, 3)  # (B, num_layers, N, hidden_size)
        
        return h_encode, None