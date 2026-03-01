import torch
import torch.nn as nn


class SpilloverCorrector(nn.Module):
    """Amortized counterfactual spillover correction module.

    3-stage architecture that models noise propagation mechanism:
      Stage 1 (Reliability): per-node score r ∈ [0,1] — "is this node's input normal?"
      Stage 2 (Propagation): cross-attention with asymmetric anomaly signals
                             — "how do anomalous nodes affect my prediction?"
      Stage 3 (Correction):  signed δ per node — "how much should I correct?"

    Backbone-agnostic: only requires temporal-pooled hidden features (B, N, d_model).
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()

        # Stage 1: Reliability Estimation
        self.reliability = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Stage 2: Anomaly Propagation
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Stage 3: Correction (zero-init last layer → initial δ=0)
        self.correction = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, h):
        """
        Args:
            h: (B, N, d_model) — temporal-pooled backbone hidden features

        Returns:
            delta: (B, N, 1) — signed correction per node
            reliability: (B, N, 1) — reliability score per node
        """
        # Stage 1: per-node reliability
        r = self.reliability(h)                       # (B, N, 1)

        # Stage 2: anomaly propagation via cross-attention
        anomaly_signal = (1 - r) * h                  # amplify unreliable nodes
        propagated, _ = self.cross_attn(
            query=h,
            key=anomaly_signal,
            value=anomaly_signal,
        )
        propagated = self.norm(h + propagated)        # residual + layernorm

        # Stage 3: per-node signed correction
        delta = self.correction(propagated)           # (B, N, 1)

        return delta, r
