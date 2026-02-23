"""
Context-aware Spatio-Temporal Encoder for traffic prediction.

This encoder learns anomaly-aware representations through contrastive pre-training.
It incorporates temporal context (time-of-day, day-of-week) and spatial context (node identity)
to generate context-dependent representations.

Reference: design.md - Section 3.1
"""

import torch
import torch.nn as nn
from typing import Optional


class ContextAwareSTEncoder(nn.Module):
    """
    Context-aware Spatio-Temporal Encoder.

    Encodes traffic time series with context embeddings (time-of-day, day-of-week, node identity)
    using a Transformer architecture.

    Args:
        input_dim: Input feature dimension (default: 1 for flow)
        d_model: Hidden dimension (default: 256)
        n_nodes: Number of nodes in the graph (dataset-specific)
        n_layers: Number of transformer encoder layers (default: 3)
        n_heads: Number of attention heads (default: 8)
        input_len: Input sequence length (default: 12)
        steps_per_day: Number of time steps per day (default: 288 for 5-min intervals)
        dropout: Dropout rate (default: 0.1)
        use_graph: Whether to use graph structure in attention (default: False)
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 256,
        n_nodes: int = 307,
        n_layers: int = 3,
        n_heads: int = 8,
        input_len: int = 12,
        steps_per_day: int = 288,
        dropout: float = 0.1,
        use_graph: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_len = input_len
        self.steps_per_day = steps_per_day
        self.use_graph = use_graph

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Context embeddings
        self.time_of_day_embed = nn.Embedding(steps_per_day, d_model)
        self.day_of_week_embed = nn.Embedding(7, d_model)
        self.node_embed = nn.Embedding(n_nodes, d_model)

        # Learnable temporal positional encoding (within window)
        self.temporal_pos = nn.Parameter(torch.randn(1, input_len, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Layer normalization for output
        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

        # Special initialization for embeddings
        nn.init.normal_(self.time_of_day_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.day_of_week_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.node_embed.weight, mean=0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_idx: Optional[torch.Tensor] = None,
        day_of_week_idx: Optional[torch.Tensor] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor of shape (B, T, N, F) - traffic flow values
            time_of_day_idx: Time of day indices (B, T) with values in [0, steps_per_day-1]
                            If None, will be inferred or set to zeros
            day_of_week_idx: Day of week indices (B,) with values in [0, 6]
                            If None, will be set to zeros
            node_idx: Node indices (N,) with values in [0, n_nodes-1]
                     If None, will be set to [0, 1, ..., N-1]

        Returns:
            z: Encoded representations of shape (B, T, N, D)
        """
        B, T, N, F = x.shape
        device = x.device

        # Default indices if not provided
        if node_idx is None:
            node_idx = torch.arange(N, device=device)

        if time_of_day_idx is None:
            time_of_day_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        if day_of_week_idx is None:
            day_of_week_idx = torch.zeros(B, dtype=torch.long, device=device)

        # 1. Project input: (B, T, N, F) -> (B, T, N, D)
        h = self.input_proj(x)

        # 2. Add context embeddings
        # Time of day: (B, T) -> (B, T, D) -> (B, T, 1, D)
        tod_emb = self.time_of_day_embed(time_of_day_idx).unsqueeze(2)

        # Day of week: (B,) -> (B, D) -> (B, 1, 1, D)
        dow_emb = self.day_of_week_embed(day_of_week_idx).unsqueeze(1).unsqueeze(2)

        # Node identity: (N,) -> (N, D) -> (1, 1, N, D)
        node_emb = self.node_embed(node_idx).unsqueeze(0).unsqueeze(0)

        # Combine all embeddings
        h = h + tod_emb + dow_emb + node_emb

        # Add learnable temporal positional encoding
        h = h + self.temporal_pos[:, :T, :, :]

        # 3. Reshape for transformer: (B, T, N, D) -> (B, T*N, D)
        h = h.view(B, T * N, -1)

        # 4. Transformer encoding
        z = self.transformer(h)

        # 5. Reshape back: (B, T*N, D) -> (B, T, N, D)
        z = z.view(B, T, N, -1)

        # 6. Output projection and normalization
        z = self.output_proj(z)
        z = self.output_norm(z)

        return z

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.d_model

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class ContextAwareSTEncoderWithPooling(ContextAwareSTEncoder):
    """
    Context-aware ST Encoder with optional pooling for downstream tasks.

    Adds pooling options for generating fixed-size representations.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 256,
        n_nodes: int = 307,
        n_layers: int = 3,
        n_heads: int = 8,
        input_len: int = 12,
        steps_per_day: int = 288,
        dropout: float = 0.1,
        use_graph: bool = False,
        pooling: str = 'none',  # 'none', 'temporal', 'spatial', 'both'
    ):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            n_nodes=n_nodes,
            n_layers=n_layers,
            n_heads=n_heads,
            input_len=input_len,
            steps_per_day=steps_per_day,
            dropout=dropout,
            use_graph=use_graph,
        )

        self.pooling = pooling

        if pooling == 'temporal':
            self.pool_proj = nn.Linear(d_model, d_model)
        elif pooling == 'spatial':
            self.pool_proj = nn.Linear(d_model, d_model)
        elif pooling == 'both':
            self.pool_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_idx: Optional[torch.Tensor] = None,
        day_of_week_idx: Optional[torch.Tensor] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with optional pooling.

        Returns:
            If pooling='none': (B, T, N, D)
            If pooling='temporal': (B, N, D)
            If pooling='spatial': (B, T, D)
            If pooling='both': (B, D)
        """
        # Get full encoder output
        z = super().forward(x, time_of_day_idx, day_of_week_idx, node_idx)

        if self.pooling == 'none':
            return z
        elif self.pooling == 'temporal':
            # Average over time dimension
            z = z.mean(dim=1)  # (B, N, D)
            z = self.pool_proj(z)
            return z
        elif self.pooling == 'spatial':
            # Average over node dimension
            z = z.mean(dim=2)  # (B, T, D)
            z = self.pool_proj(z)
            return z
        elif self.pooling == 'both':
            # Average over both time and node dimensions
            z = z.mean(dim=(1, 2))  # (B, D)
            z = self.pool_proj(z)
            return z
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
