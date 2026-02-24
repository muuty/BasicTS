"""
STHCN: Spatio-Temporal Hypergraph Convolutional Network.
Core encoding module for GPT-ST model.
"""

import torch
import torch.nn as nn

from .modules import HyperTem, Cap, TimeFeature, TimeFeatureSPG


class STHCN(nn.Module):
    """Spatio-Temporal Hypergraph Convolutional Network.

    Combines temporal hypergraph convolution (hyperTem) with
    spatial capsule routing (cap) for joint spatio-temporal learning.
    """

    def __init__(
        self,
        num_nodes: int,
        input_base_dim: int,
        hidden_dim: int,
        horizon: int,
        embed_dim: int,
        embed_dim_spa: int,
        HS: int,
        HT: int,
        HT_Tem: int,
        num_route: int,
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph
            input_base_dim: Input feature dimension (e.g., 1 for flow)
            hidden_dim: Hidden dimension
            horizon: Number of timesteps (sequence length)
            embed_dim: Embedding dimension for temporal features
            embed_dim_spa: Embedding dimension for spatial features
            HS: Number of spatial hyperedge heads
            HT: Number of temporal hyperedge heads
            HT_Tem: Number of temporal hypergraph heads
            num_route: Number of routing iterations in capsule network
        """
        super(STHCN, self).__init__()

        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.embed_dim_spa = embed_dim_spa
        self.HS = HS
        self.HT = HT
        self.HT_Tem = HT_Tem
        self.num_route = num_route

        # Learnable node embeddings
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim), requires_grad=True
        )
        self.node_embeddings_spg = nn.Parameter(
            torch.randn(num_nodes, embed_dim), requires_grad=True
        )

        # Temporal hypergraph layers
        self.hyperTem1 = HyperTem(horizon, num_nodes, hidden_dim, hidden_dim, embed_dim, HT_Tem)
        self.hyperTem2 = HyperTem(horizon, num_nodes, hidden_dim, hidden_dim, embed_dim, HT_Tem)
        self.hyperTem3 = HyperTem(horizon, num_nodes, hidden_dim, hidden_dim, embed_dim, HT_Tem)
        self.hyperTem4 = HyperTem(horizon, num_nodes, hidden_dim, hidden_dim, embed_dim, HT_Tem)

        # Time feature encoders
        self.time_feature1 = TimeFeature(embed_dim)
        self.time_feature1_ = TimeFeature(embed_dim_spa)
        self.time_feature2 = TimeFeatureSPG(embed_dim_spa)

        # Capsule routing layers
        self.cap1 = Cap(hidden_dim, num_nodes, horizon, embed_dim, embed_dim_spa, HS, HT, num_route)
        self.cap2 = Cap(hidden_dim, num_nodes, horizon, embed_dim, embed_dim_spa, HS, HT, num_route)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.node_embeddings_spg)

    def forward(self, source: torch.Tensor, x_in: torch.Tensor) -> tuple:
        """
        Args:
            source: Raw input with temporal features [B, T, N, D]
                   where D includes input_base_dim + temporal features (tod, dow)
            x_in: Encoded input [B, T, N, hidden_dim]

        Returns:
            xt4: Output embedding [B, T, N, hidden_dim]
            HS1: Spatial hyperedge weights from first cap layer
            HS3: Spatial hyperedge weights from second cap layer
        """
        # Extract temporal indices
        day_index = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
        week_index = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]

        # Generate time embeddings
        time_eb = self.time_feature1(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        teb = self.time_feature1_(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        time_eb_spg = self.time_feature2(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)

        # First block: hyperTem1 -> cap1 -> hyperTem2
        xt1 = self.hyperTem1(x_in, self.node_embeddings, time_eb)
        x_hyperTem_gnn1, HS1, HT1 = self.cap1(xt1, self.node_embeddings_spg, time_eb_spg, teb)
        xt2 = self.hyperTem2(x_hyperTem_gnn1, self.node_embeddings, time_eb)

        # Second block: hyperTem3 -> cap2 -> hyperTem4
        xt3 = self.hyperTem3(xt2, self.node_embeddings, time_eb)
        x_hyperTem_gnn3, HS3, HT3 = self.cap2(xt3, self.node_embeddings_spg, time_eb_spg, teb)
        xt4 = self.hyperTem4(x_hyperTem_gnn3, self.node_embeddings, time_eb)

        return xt4, HS1, HS3
