#!/usr/bin/env python3
"""GRU Seq2Seq Split Learning 클라이언트 모델 (GraphNet은 서버에서 처리)"""

import numpy as np
import torch
import torch.nn as nn

from ...base_models import BaseClientModel


class GRUSeq2SeqSplitClientModel(BaseClientModel):
    """
    GRU Seq2Seq Split Learning 클라이언트 모델
    - GRU Encoder: 로컬 시계열 데이터 인코딩
    - GRU Decoder: 서버에서 받은 graph_encoding과 결합하여 예측 수행
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_size: int = 128,
        output_dim: int = 1,
        gru_num_layers: int = 2,
        dropout: float = 0.0,
        cl_decay_steps: int = 2000,
        use_curriculum_learning: bool = True,
        seq_len: int = 12,
        horizon: int = 12,
    ):
        super().__init__(num_nodes, hidden_size)
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.gru_num_layers = gru_num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # GRU Encoder
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=False  # (Seq, Batch, Feat)
        )
        
        # GRU Decoder (hidden size doubled for concatenation: encoder_h + graph_h)
        self.decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=2 * hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output projection
        self.out_net = nn.Linear(2 * hidden_size, output_dim)
    
    def _compute_sampling_threshold(self, batches_seen):
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward_encoder(self, history_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_data: (B, L, N, C)
        Returns:
            h_encode: (num_layers, B*N, hidden_size)
        """
        B, L, N, C = history_data.shape
        
        # Reshape: (B, L, N, C) -> (L, B, N, C) -> (L, B*N, C)
        x_input = history_data.permute(1, 0, 2, 3).reshape(L, B * N, C)
        
        # Encode
        _, h_encode = self.encoder(x_input)
        
        return h_encode
    
    def forward_decoder(
        self,
        history_data: torch.Tensor,
        h_encode: torch.Tensor,
        graph_encoding: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            history_data: (B, L, N, C) - 마지막 입력값 추출용
            h_encode: (num_layers, B*N, hidden_size) - Encoder hidden state
            graph_encoding: (num_layers, B*N, hidden_size) - Server result
            future_data: (B, horizon, N, C) - Teacher forcing용
        Returns:
            prediction: (B, horizon, N, output_dim)
        """
        B, L, N, C = history_data.shape
        
        # Prepare Input (Last step of history)
        # (B, L, N, C) -> (L, B*N, C)
        x_input_all = history_data.permute(1, 0, 2, 3).reshape(L, B * N, C)
        last_input = x_input_all[-1:]  # (1, B*N, C)
        
        # Concatenate Hidden States
        # h_decode: (num_layers, B*N, 2*hidden_size)
        h_decode = torch.cat([h_encode, graph_encoding], dim=-1)
        
        last_hidden = h_decode
        out_steps = []
        
        # Prepare GT for Teacher Forcing
        if future_data is not None:
            # (B, H, N, C) -> (H, B*N, C)
            y_gt = future_data[..., :self.input_dim].permute(1, 0, 2, 3).reshape(self.horizon, B * N, -1)
        
        # Decoding Loop
        for t in range(self.horizon):
            out_hidden, last_hidden = self.decoder(last_input, last_hidden)
            out = self.out_net(out_hidden)  # (1, B*N, output_dim)
            out_steps.append(out)
            
            # Curriculum Learning / Teacher Forcing
            if self.training and self.use_curriculum_learning and future_data is not None:
                p_gt = self._compute_sampling_threshold(batch_seen)
                use_gt = np.random.uniform(0, 1) < p_gt
                
                if use_gt:
                    next_in = y_gt[t:t+1]
                else:
                    # Pad if output_dim < input_dim
                    if out.shape[-1] < self.input_dim:
                        padding = torch.zeros(1, B * N, self.input_dim - out.shape[-1], device=out.device)
                        next_in = torch.cat([out, padding], dim=-1)
                    else:
                        next_in = out[..., :self.input_dim]
                last_input = next_in
            else:
                # Inference Mode
                if out.shape[-1] < self.input_dim:
                    padding = torch.zeros(1, B * N, self.input_dim - out.shape[-1], device=out.device)
                    last_input = torch.cat([out, padding], dim=-1)
                else:
                    last_input = out[..., :self.input_dim]
        
        # Stack & Reshape
        # (horizon, 1, B*N, output_dim) -> (horizon, B, N, output_dim)
        out = torch.cat(out_steps, dim=0)
        out = out.view(self.horizon, B, N, self.output_dim)
        out = out.permute(1, 0, 2, 3)  # (B, horizon, N, output_dim)
        
        return out

