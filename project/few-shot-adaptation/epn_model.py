"""Embedding Predictor Network (EPN)

Predicts Node-Specific Parameters (NSP) from traffic pattern + graph context.
Trained via synthetic episodes: mask existing nodes, predict their learned NSP.

For STAEformer: NSP = adaptive_embedding, shape (T, N, D) per model, (T, D) per node.
"""

import torch
import torch.nn as nn


class TrafficPatternEncoder(nn.Module):
    """Encode a node's traffic time series into a fixed-size feature vector.

    Input: (batch, budget_steps, num_channels)
    Output: (batch, d_hidden)
    """

    def __init__(self, num_channels=5, d_hidden=64):
        super().__init__()
        # 1D conv over time axis, then global average pooling
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # handles variable-length input

    def forward(self, x):
        """x: (batch, time_steps, channels)"""
        x = x.transpose(1, 2)  # (batch, channels, time_steps)
        x = self.conv(x)       # (batch, d_hidden, time_steps)
        x = self.pool(x)       # (batch, d_hidden, 1)
        return x.squeeze(-1)   # (batch, d_hidden)


class GraphContextEncoder(nn.Module):
    """Encode a node's graph neighborhood into a fixed-size feature vector.

    Uses adjacency-weighted mean of neighbor NSPs.

    Input: neighbor_nsps (batch, max_neighbors, nsp_dim), adj_weights (batch, max_neighbors)
    Output: (batch, d_hidden)
    """

    def __init__(self, nsp_dim, d_hidden=64):
        super().__init__()
        self.proj = nn.Linear(nsp_dim, d_hidden)

    def forward(self, neighbor_nsps, adj_weights):
        """
        neighbor_nsps: (batch, max_neighbors, nsp_dim) - flattened NSP of each neighbor
        adj_weights: (batch, max_neighbors) - adjacency weights (0 for padding)
        """
        # Normalize weights (masked softmax)
        mask = (adj_weights > 0).float()
        weights = adj_weights * mask
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum  # (batch, max_neighbors)

        # Weighted mean of neighbor NSPs
        weighted = (neighbor_nsps * weights.unsqueeze(-1)).sum(dim=1)  # (batch, nsp_dim)
        return self.proj(weighted)  # (batch, d_hidden)


class EmbeddingPredictorNetwork(nn.Module):
    """EPN: Predict a node's NSP from its traffic pattern and graph context.

    For STAEformer: nsp_dim = T * D = 12 * 24 = 288
    """

    def __init__(self, nsp_dim, num_channels=5, d_hidden=64):
        super().__init__()
        self.nsp_dim = nsp_dim

        self.pattern_encoder = TrafficPatternEncoder(num_channels, d_hidden)
        self.graph_encoder = GraphContextEncoder(nsp_dim, d_hidden)

        self.fusion = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 2),
            nn.ReLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, nsp_dim),
        )

    def forward(self, traffic_data, neighbor_nsps, adj_weights):
        """
        traffic_data: (batch, time_steps, channels)
        neighbor_nsps: (batch, max_neighbors, nsp_dim)
        adj_weights: (batch, max_neighbors)

        Returns: predicted_nsp (batch, nsp_dim)
        """
        pattern_feat = self.pattern_encoder(traffic_data)    # (batch, d_hidden)
        graph_feat = self.graph_encoder(neighbor_nsps, adj_weights)  # (batch, d_hidden)
        fused = torch.cat([pattern_feat, graph_feat], dim=-1)  # (batch, d_hidden*2)
        return self.fusion(fused)  # (batch, nsp_dim)

    def forward_pattern_only(self, traffic_data):
        """Ablation: predict NSP from traffic pattern only (no graph context)."""
        pattern_feat = self.pattern_encoder(traffic_data)
        # Zero graph context
        graph_feat = torch.zeros_like(pattern_feat)
        fused = torch.cat([pattern_feat, graph_feat], dim=-1)
        return self.fusion(fused)

    def forward_graph_only(self, neighbor_nsps, adj_weights):
        """Ablation: predict NSP from graph context only (no traffic data)."""
        graph_feat = self.graph_encoder(neighbor_nsps, adj_weights)
        # Zero pattern
        pattern_feat = torch.zeros_like(graph_feat)
        fused = torch.cat([pattern_feat, graph_feat], dim=-1)
        return self.fusion(fused)
