"""
Gated Residual MLP: Node-adaptive residual correction.

Unlike ResidualMLP which applies uniform correction to all nodes,
this model has a gate network that learns WHICH nodes need correction.

    gate ≈ 0 → backbone prediction preserved (normal nodes)
    gate ≈ 1 → full correction applied (anomalous/worst nodes)

Final: ŷ = ŷ_backbone + gate * Δŷ
"""
import torch
import torch.nn as nn


class GatedResidualMLP(nn.Module):
    """Node-adaptive residual correction with learned gating.

    Args:
        input_dim: Input feature dimension (raw or representation).
        in_steps: Number of input timesteps.
        out_steps: Number of output timesteps to predict.
        hidden_dim: Hidden dimension of the MLPs.
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

        # Correction branch: predicts Δŷ
        self.correction_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.correction_temporal = nn.Linear(in_steps, out_steps)
        self.correction_out = nn.Linear(hidden_dim, 1)

        # Gate branch: predicts per-node gate [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_temporal = nn.Linear(in_steps, out_steps)
        self.gate_out = nn.Linear(hidden_dim, 1)

    def forward(self, history_data, future_data=None, batch_seen=None,
                epoch=None, train=False, **kwargs):
        """
        Args:
            history_data: [B, T_in, N, input_dim]
        Returns:
            dict with 'prediction': [B, T_out, N, 1] gated residual (gate * Δŷ)
        """
        # Correction branch
        c = self.correction_mlp(history_data)     # [B, T_in, N, hidden]
        c = c.permute(0, 2, 3, 1)                 # [B, N, hidden, T_in]
        c = self.correction_temporal(c)            # [B, N, hidden, T_out]
        c = c.permute(0, 3, 1, 2)                 # [B, T_out, N, hidden]
        delta_y = self.correction_out(c)           # [B, T_out, N, 1]

        # Gate branch
        g = self.gate_mlp(history_data)            # [B, T_in, N, hidden]
        g = g.permute(0, 2, 3, 1)                 # [B, N, hidden, T_in]
        g = self.gate_temporal(g)                  # [B, N, hidden, T_out]
        g = g.permute(0, 3, 1, 2)                 # [B, T_out, N, hidden]
        gate = torch.sigmoid(self.gate_out(g))     # [B, T_out, N, 1] in (0, 1)

        return {'prediction': gate * delta_y}
