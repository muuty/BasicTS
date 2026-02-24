"""
Dual-Head Forecaster: Separate prediction paths for context and self representations.

Key idea: Don't let downstream model re-mix disentangled representations.
- z_context → Context Head → pred_context (neighbor-based prediction)
- z_self → Self Head → pred_self (self-based prediction)
- Learnable gate combines predictions per node

This preserves disentanglement all the way to the output.
"""
import torch
import torch.nn as nn
import math


class TemporalAttention(nn.Module):
    """Simple temporal attention for sequence modeling."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D] -> [B*N, T, D]
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))

        # [B*N, T, D] -> [B, T, N, D]
        x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return x


class PredictionHead(nn.Module):
    """Prediction head: temporal attention + MLP projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        in_steps: int,
        out_steps: int,
        num_nodes: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # Temporal attention
        self.temporal_attn = TemporalAttention(input_dim, nhead, dropout)

        # Temporal aggregation + projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(input_dim * in_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps * output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T, N, D] representation
        Returns:
            pred: [B, out_steps, N, output_dim] predictions
        """
        B, T, N, D = z.shape

        # Temporal attention (within each node)
        z = self.temporal_attn(z)  # [B, T, N, D]

        # Flatten temporal dimension and project
        z = z.permute(0, 2, 1, 3).reshape(B, N, T * D)  # [B, N, T*D]
        pred = self.temporal_proj(z)  # [B, N, out_steps * output_dim]

        # Reshape to output format
        pred = pred.reshape(B, N, self.out_steps, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)  # [B, out_steps, N, output_dim]

        return pred


class NodeAdaptiveGate(nn.Module):
    """Learnable per-node gate to balance context vs self predictions."""

    def __init__(self, num_nodes: int, init_bias: float = 0.0):
        super().__init__()
        # Learnable gate per node
        self.gate_logits = nn.Parameter(torch.zeros(num_nodes) + init_bias)

    def forward(self, pred_context: torch.Tensor, pred_self: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_context: [B, T, N, D] context-based predictions
            pred_self: [B, T, N, D] self-based predictions
        Returns:
            pred: [B, T, N, D] combined predictions
        """
        # gate: [N] -> [1, 1, N, 1]
        gate = torch.sigmoid(self.gate_logits).view(1, 1, -1, 1)

        # Combine: gate=1 means use context, gate=0 means use self
        pred = gate * pred_context + (1 - gate) * pred_self

        return pred

    def get_gate_values(self) -> torch.Tensor:
        """Return gate values for analysis."""
        return torch.sigmoid(self.gate_logits)


class DualHeadForecaster(nn.Module):
    """
    Dual-Head Forecaster with disentangled prediction paths.

    Architecture:
        Input → DisentangledEncoder → z_context, z_self
                                           ↓           ↓
                                    Context Head   Self Head
                                           ↓           ↓
                                    pred_context   pred_self
                                           ↓           ↓
                                        Node-Adaptive Gate
                                               ↓
                                         Final Prediction

    Key benefits:
    - Preserves disentanglement through prediction
    - Node-specific balance between context and self
    - Outlier nodes can rely on self-representation
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        context_dim: int,  # d_model from DisentangledEncoder
        self_dim: int,     # d_model from DisentangledEncoder
        hidden_dim: int = 256,
        output_dim: int = 1,
        nhead: int = 4,
        dropout: float = 0.1,
        gate_init_bias: float = 0.0,  # 0 = balanced, positive = prefer context
        **kwargs  # ignore extra args
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.context_dim = context_dim
        self.self_dim = self_dim

        # Separate prediction heads
        self.context_head = PredictionHead(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            in_steps=in_steps,
            out_steps=out_steps,
            num_nodes=num_nodes,
            nhead=nhead,
            dropout=dropout,
        )

        self.self_head = PredictionHead(
            input_dim=self_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            in_steps=in_steps,
            out_steps=out_steps,
            num_nodes=num_nodes,
            nhead=nhead,
            dropout=dropout,
        )

        # Node-adaptive gate
        self.gate = NodeAdaptiveGate(num_nodes, init_bias=gate_init_bias)

    def forward(
        self,
        history_data: torch.Tensor,
        z_context: torch.Tensor = None,
        z_self: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            history_data: [B, T, N, D] - if z_context/z_self not provided,
                          expects concatenated [z_context, z_self] format
            z_context: [B, T, N, context_dim] - optional, explicit context representation
            z_self: [B, T, N, self_dim] - optional, explicit self representation
        Returns:
            pred: [B, out_steps, N, output_dim]
        """
        # If explicit representations not provided, split from history_data
        if z_context is None or z_self is None:
            # Assume history_data is [z_context || z_self || tod || dow]
            # context_dim and self_dim should match
            z_context = history_data[..., :self.context_dim]
            z_self = history_data[..., self.context_dim:self.context_dim + self.self_dim]

        # Separate predictions
        pred_context = self.context_head(z_context)  # [B, out_steps, N, 1]
        pred_self = self.self_head(z_self)           # [B, out_steps, N, 1]

        # Combine with node-adaptive gate
        pred = self.gate(pred_context, pred_self)    # [B, out_steps, N, 1]

        return pred

    def get_gate_analysis(self) -> dict:
        """Return gate analysis for debugging."""
        gate_values = self.gate.get_gate_values().detach().cpu()
        return {
            'gate_values': gate_values,
            'mean_gate': gate_values.mean().item(),
            'std_gate': gate_values.std().item(),
            'min_gate': gate_values.min().item(),
            'max_gate': gate_values.max().item(),
            'nodes_prefer_self': (gate_values < 0.5).sum().item(),
            'nodes_prefer_context': (gate_values >= 0.5).sum().item(),
        }
