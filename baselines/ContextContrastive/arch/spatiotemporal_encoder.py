"""
Spatio-Temporal Encoder with Temporal Self-Attention + GAT-based Spatial Encoding.

Architecture:
    Input → Temporal Encoder → Spatial Encoder (GAT) → Output

Features:
    - Top-K sparsification for memory-efficient GAT
    - Causal temporal attention
    - Multi-head graph attention for spatial relationships
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Transformer-based temporal encoder with causal masking."""

    def __init__(self, c_in: int, d_model: int, num_layers: int = 2, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(c_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            z: [B, T, N, D] - encoded representations
        """
        B, T, N, C = x.shape

        # Reshape: treat each node independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # Project to d_model
        x = self.input_proj(x)

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        # Encode with causal attention
        x = self.transformer(x, mask=causal_mask, is_causal=False)

        # Reshape back
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        return x


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer (GAT).

    Memory-efficient implementation that works with sparse adjacency.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, D] node features
            adj: [N, N] sparse adjacency matrix (top-k sparsified)
        Returns:
            h': [B, N, D] updated node features
        """
        B, N, D = h.shape

        # Linear transformation
        Wh = self.W(h)  # [B, N, out_features]

        # Attention mechanism
        # Compute attention coefficients
        Wh1 = Wh @ self.a[:, :self.out_features].T  # [B, N, 1]
        Wh2 = Wh @ self.a[:, self.out_features:].T  # [B, N, 1]

        # Broadcast and add: e_ij = a^T [Wh_i || Wh_j]
        e = Wh1 + Wh2.transpose(-1, -2)  # [B, N, N]
        e = self.leakyrelu(e)

        # Mask with adjacency (only attend to neighbors)
        # adj: [N, N], need to expand for batch
        mask = (adj == 0)
        e = e.masked_fill(mask.unsqueeze(0), float('-inf'))

        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)  # [B, N, N]
        attention = self.dropout(attention)

        # Handle NaN from softmax (isolated nodes)
        attention = torch.nan_to_num(attention, nan=0.0)

        # Aggregate neighbor features
        h_prime = attention @ Wh  # [B, N, out_features]

        return h_prime


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention."""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        if concat:
            assert out_features % num_heads == 0
            head_dim = out_features // num_heads
        else:
            head_dim = out_features

        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, head_dim, dropout)
            for _ in range(num_heads)
        ])

        self.out_features = out_features

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, D] node features
            adj: [N, N] sparse adjacency matrix
        Returns:
            h': [B, N, out_features]
        """
        head_outputs = [attn(h, adj) for attn in self.attention_heads]

        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            return torch.stack(head_outputs, dim=0).mean(dim=0)


class SpatialEncoder(nn.Module):
    """
    GAT-based spatial encoder with top-k sparsification.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 1,
        k_neighbors: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k_neighbors
        self.d_model = d_model
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList([
            MultiHeadGAT(d_model, d_model, num_heads, dropout, concat=True)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cache for sparse adjacency
        self._sparse_adj = None
        self._adj_device = None

    def sparsify_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Keep only top-k neighbors per node.

        Args:
            adj: [N, N] dense adjacency matrix
        Returns:
            sparse_adj: [N, N] sparse adjacency with only top-k neighbors
        """
        N = adj.shape[0]
        k = min(self.k, N - 1)

        # Get top-k indices and values per row
        topk_vals, topk_indices = adj.topk(k, dim=-1)

        # Create sparse adjacency
        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(-1, topk_indices, topk_vals)

        # Make symmetric (undirected graph)
        sparse_adj = (sparse_adj + sparse_adj.T) / 2

        return sparse_adj

    def get_sparse_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Get or compute sparse adjacency (cached)."""
        if self._sparse_adj is None or self._adj_device != adj.device:
            self._sparse_adj = self.sparsify_adj(adj)
            self._adj_device = adj.device
        return self._sparse_adj

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D] - temporally encoded features
            adj: [N, N] - adjacency matrix
        Returns:
            z: [B, T, N, D] - spatially encoded features
        """
        B, T, N, D = x.shape

        # Get sparse adjacency
        sparse_adj = self.get_sparse_adj(adj)

        # Process each timestep
        outputs = []
        for t in range(T):
            h = x[:, t, :, :]  # [B, N, D]

            # Apply GAT layers with residual
            for gat, ln in zip(self.gat_layers, self.layer_norms):
                h_new = gat(h, sparse_adj)
                h_new = self.dropout(h_new)
                h = ln(h + h_new)  # Residual connection

            outputs.append(h)

        # Stack timesteps
        z = torch.stack(outputs, dim=1)  # [B, T, N, D]

        return z


class SpatioTemporalEncoder(nn.Module):
    """
    Spatio-Temporal Encoder: Temporal Self-Attention → GAT Spatial Encoding.

    Architecture:
        Input [B, T, N, C]
            → Temporal Encoder (causal self-attention)
            → Spatial Encoder (GAT with top-k)
            → Output [B, T, N, D]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        spatial_layers: int = 1,
        spatial_heads: int = 4,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            c_in=input_dim,
            d_model=d_model,
            num_layers=temporal_layers,
            nhead=temporal_heads,
            dropout=dropout
        )

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            d_model=d_model,
            num_heads=spatial_heads,
            num_layers=spatial_layers,
            k_neighbors=k_neighbors,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
            adj: [N, N] - adjacency matrix (required for spatial encoding)
        Returns:
            z: [B, T, N, D] - spatio-temporal encoded representations
        """
        # Temporal encoding
        z = self.temporal_encoder(x)  # [B, T, N, D]

        # Spatial encoding (if adjacency provided)
        if adj is not None:
            z = self.spatial_encoder(z, adj)  # [B, T, N, D]

        return z
