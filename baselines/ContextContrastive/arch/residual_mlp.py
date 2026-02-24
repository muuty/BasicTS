"""
Residual MLP: Lightweight post-processor for residual learning.

Predicts a residual correction Δŷ to add to a frozen backbone's prediction.
Input can be raw features or disentangled representations from a frozen encoder.

Architecture:
    Input → Backbone (frozen) → ŷ_backbone
    Input → [optional Encoder] → ResidualMLP → Δŷ
    Final: ŷ = ŷ_backbone + Δŷ
"""
import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Predicts residual correction from raw input or representations.

    Args:
        input_dim: Input feature dimension.
                   - Raw input: e.g. 3 (flow, tod, dow)
                   - Representation: e.g. 128 (concat z_context + z_self)
        in_steps: Number of input timesteps.
        out_steps: Number of output timesteps to predict.
        hidden_dim: Hidden dimension of the MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 128,
        in_steps: int = 12,
        out_steps: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Per-timestep feature extraction
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal projection: in_steps -> out_steps
        self.temporal_proj = nn.Linear(in_steps, out_steps)

        # Output projection: hidden -> 1 (scalar residual per node per timestep)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, history_data, future_data=None, batch_seen=None,
                epoch=None, train=False, **kwargs):
        """
        Args:
            history_data: [B, T_in, N, input_dim]
        Returns:
            dict with 'prediction': [B, T_out, N, 1] residual Δŷ
        """
        z = self.feature_mlp(history_data)    # [B, T_in, N, hidden]
        z = z.permute(0, 2, 3, 1)            # [B, N, hidden, T_in]
        z = self.temporal_proj(z)             # [B, N, hidden, T_out]
        z = z.permute(0, 3, 1, 2)            # [B, T_out, N, hidden]
        delta_y = self.output_proj(z)         # [B, T_out, N, 1]

        return {'prediction': delta_y}
