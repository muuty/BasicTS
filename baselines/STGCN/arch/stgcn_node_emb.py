"""STGCN with feature-level node embedding.

Adds a learnable node embedding (N, d) that gets concatenated to input features
before entering the ST-Conv blocks. This provides a per-node adaptive feature
that the original STGCN lacks.

Used for concept drift PB (Pattern Bank) experiments:
- Train base model with node_emb for each year
- Apply PB adapter to node_emb for cross-year adaptation
"""
import torch
import torch.nn as nn
from typing import List

from .encoder import STGCNEncoder
from .decoder import STGCNDecoder


class STGCNNodeEmb(nn.Module):
    def __init__(
        self,
        Ks: int,
        Kt: int,
        blocks: List[List[int]],
        T: int,
        num_nodes: int,
        act_func: str,
        graph_conv_type: str,
        adj_matrix: torch.Tensor,
        bias: bool,
        droprate: float,
        node_emb_dim: int = 24,
    ) -> None:
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # Adjust input channel to account for node embedding
        input_channel = blocks[0][0] + node_emb_dim

        num_st_blocks = 2
        Ko = T - num_st_blocks * 2 * (Kt - 1)

        self.encoder = STGCNEncoder(
            Kt=Kt,
            Ks=Ks,
            num_nodes=num_nodes,
            last_block_channel=input_channel,
            channels=blocks[1],
            channels_2=blocks[2],
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            adj_matrix=adj_matrix,
            bias=bias,
            droprate=droprate,
        )

        self.decoder = STGCNDecoder(
            Ko=Ko,
            last_block_channel=blocks[2][-1],
            output_channels=blocks[-2],
            out_steps=blocks[-1][0],
            num_nodes=num_nodes,
            act_func=act_func,
            bias=bias,
            droprate=droprate,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        B, T, N, C = history_data.shape

        # Concat node embedding to input features
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x = torch.cat([history_data, node_emb], dim=-1)  # (B, T, N, C+d)

        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x
