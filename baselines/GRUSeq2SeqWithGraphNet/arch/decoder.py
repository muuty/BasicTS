# decoder.py
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class GRUDecoder(nn.Module):
    """
    GRU Decoder: Hidden states를 예측값으로 디코딩
    
    Curriculum Learning 지원
    
    Input: (B, num_layers, N, 2*hidden_size) - encoder + spatial 결합
    Output: (B, horizon, N, output_dim)
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 12,
        dropout: float = 0.0,
        cl_decay_steps: int = 1000,
        use_curriculum_learning: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # Decoder GRU (doubled hidden for encoder + spatial)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=2 * hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
        )
        
        # Output projection
        self.out_proj = nn.Linear(2 * hidden_size, output_dim)
    
    def _compute_sampling_threshold(self, batches_seen: int) -> float:
        """Curriculum learning threshold"""
        if self.cl_decay_steps == 0:
            return 0
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        last_input: Optional[torch.Tensor] = None,
        future_data: Optional[torch.Tensor] = None,
        batch_seen: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, num_layers, N, 2*hidden_size) - combined hidden states
            last_input: (B, N, input_dim) - last input from encoder (optional)
            future_data: (B, horizon, N, C) - for curriculum learning (optional)
            batch_seen: for curriculum learning
        Returns:
            predictions: (B, horizon, N, output_dim)
        """
        B, L, N, H = x.shape
        device = x.device
        
        # (B, num_layers, N, 2*hidden_size) -> (num_layers, B*N, 2*hidden_size)
        h_decode = x.permute(1, 0, 2, 3).reshape(L, B * N, H)
        
        # Initialize last_input if not provided
        if last_input is None:
            last_input = torch.zeros(1, B * N, self.input_dim, device=device)
        else:
            # (B, N, input_dim) -> (1, B*N, input_dim)
            last_input = last_input.reshape(1, B * N, self.input_dim)
        
        # Prepare future data for curriculum learning
        if future_data is not None:
            y_gt = future_data[..., :self.input_dim]
            y_gt = y_gt.permute(1, 0, 2, 3).reshape(self.horizon, B * N, -1)
        
        # Autoregressive decoding
        out_steps = []
        last_hidden = h_decode
        
        for t in range(self.horizon):
            out_hidden, last_hidden = self.gru(last_input, last_hidden)
            out = self.out_proj(out_hidden)  # (1, B*N, output_dim)
            out_steps.append(out)
            
            # Prepare next input
            if self.training and self.use_curriculum_learning and future_data is not None:
                p_gt = self._compute_sampling_threshold(batch_seen)
                if np.random.uniform(0, 1) < p_gt:
                    last_input = y_gt[t:t+1]
                else:
                    last_input = self._pad_output(out, B * N, device)
            else:
                last_input = self._pad_output(out, B * N, device)
        
        # Combine outputs
        out = torch.cat(out_steps, dim=0)  # (horizon, B*N, output_dim)
        out = out.view(self.horizon, B, N, self.output_dim)
        out = out.permute(1, 0, 2, 3)  # (B, horizon, N, output_dim)
        
        return out
    
    def _pad_output(self, out: torch.Tensor, bn: int, device: torch.device) -> torch.Tensor:
        """Pad output to input_dim if needed"""
        if out.shape[-1] < self.input_dim:
            padding = torch.zeros(
                1, bn, self.input_dim - out.shape[-1],
                device=device
            )
            return torch.cat([out, padding], dim=-1)
        return out[..., :self.input_dim]