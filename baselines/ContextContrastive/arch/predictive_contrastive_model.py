"""
Predictive Contrastive Pretrain Model.

Learns representations by predicting future from past:
    z_past = encoder(x[:, :T//2])
    z_future = encoder(x[:, T//2:])
    loss = -cosine_sim(predictor(z_past), stop_grad(z_future))

Key design choices:
- Single encoder (no disentanglement) - directly useful for forecasting
- Asymmetric predictor + stop-gradient prevents collapse (SimSiam-style)
- Predictive objective aligns with downstream forecasting task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .encoders import TransformerEncoder


class PredictiveContrastiveModel(nn.Module):
    """
    Predictive contrastive pretraining model.

    Architecture:
        Input [B, T, N, C]
            ├── x_past  = x[:, :T//2]  ──► encoder ──► z_past  ──► predictor ──► z_pred
            └── x_future = x[:, T//2:] ──► encoder ──► z_future (stop-gradient target)

        Loss = 2 - 2 * cosine_sim(z_pred, sg(z_future))
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        predictor_hidden: int = 128,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model

        # Shared encoder for past and future
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )

        # Asymmetric predictor (prevents collapse without EMA)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Args:
            x: [B, T, N, C] input tensor

        Returns:
            Dict with z_pred and z_target for loss computation
        """
        B, T, N, C = x.shape
        T_split = T // 2

        x_past = x[:, :T_split]      # [B, T//2, N, C]
        x_future = x[:, T_split:]    # [B, T//2, N, C]

        # Encode both halves with shared encoder
        z_past = self.encoder.encode(x_past)       # [B, T//2, N, D]
        z_future = self.encoder.encode(x_future)   # [B, T//2, N, D]

        # Pool temporal dimension
        z_past_pooled = z_past.mean(dim=1)         # [B, N, D]
        z_future_pooled = z_future.mean(dim=1)     # [B, N, D]

        # Predict future from past
        z_pred = self.predictor(z_past_pooled)     # [B, N, D]

        return {
            'z_pred': z_pred,
            'z_target': z_future_pooled,
        }

    def compute_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss with stop-gradient on target.

        Returns value in [0, 4] range (2 - 2*cos_sim).
        """
        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target.detach(), dim=-1)  # stop-gradient
        return 2 - 2 * (z_pred * z_target).sum(dim=-1).mean()

    def get_encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()
