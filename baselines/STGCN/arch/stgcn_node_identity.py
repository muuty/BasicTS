"""
STGCN with Node Identity Preservation

Experiment: Add learnable node embedding after message passing
to preserve node-specific information that may be smoothed out.
"""

import torch
import torch.nn as nn
from typing import List

from .stgcn_layers import STConvBlock, OutputBlock


class STGCNNodeIdentity(nn.Module):
    """
    STGCN + Node Identity Embedding

    After message passing, add learnable node embeddings to preserve
    node-specific information.

    Args:
        node_embed_dim: Dimension of node embedding (0 to disable)
        node_embed_mode: 'add' or 'concat'
    """

    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate,
                 node_embed_dim: int = 32, node_embed_mode: str = 'add'):
        super(STGCNNodeIdentity, self).__init__()
        self.gso = gso
        self.T = T
        self.n_vertex = n_vertex
        self.node_embed_dim = node_embed_dim
        self.node_embed_mode = node_embed_mode

        # ST-Conv blocks
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, bias, droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."

        # Node identity embedding
        if node_embed_dim > 0:
            # Get the channel dimension after ST blocks
            st_out_channels = blocks[-3][-1]

            self.node_embedding = nn.Embedding(n_vertex, node_embed_dim)

            if node_embed_mode == 'add':
                # Project node embedding to match ST output channels
                self.node_proj = nn.Linear(node_embed_dim, st_out_channels)
                output_channels = st_out_channels
            else:  # concat
                output_channels = st_out_channels + node_embed_dim
                # Adjust output block input if concat
                self.channel_adjust = nn.Conv2d(output_channels, st_out_channels, 1)
        else:
            self.node_embedding = None

        # Output block
        self.output = OutputBlock(
            Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, bias, droprate)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data: [B, L, N, C]
        Returns:
            dict with 'prediction' [B, L, N, 1] and 'repr' [B, C, L', N]
        """
        B, L, N, C = history_data.shape

        # [B, L, N, C] -> [B, C, L, N]
        x = history_data.permute(0, 3, 1, 2).contiguous()

        # ST-Conv blocks (with message passing)
        x = self.st_blocks(x)  # [B, C', L', N]

        # Add node identity after message passing
        if self.node_embedding is not None:
            node_ids = torch.arange(N, device=x.device)
            node_emb = self.node_embedding(node_ids)  # [N, node_embed_dim]

            if self.node_embed_mode == 'add':
                # Project and add
                node_emb_proj = self.node_proj(node_emb)  # [N, C']
                # Broadcast: [N, C'] -> [1, C', 1, N]
                node_emb_proj = node_emb_proj.T.unsqueeze(0).unsqueeze(2)
                x = x + node_emb_proj
            else:  # concat
                # [N, node_embed_dim] -> [1, node_embed_dim, 1, N]
                node_emb = node_emb.T.unsqueeze(0).unsqueeze(2)
                # Expand to match x's batch and temporal dims
                node_emb = node_emb.expand(B, -1, x.size(2), -1)
                x = torch.cat([x, node_emb], dim=1)
                x = self.channel_adjust(x)

        repr = x.clone()

        # Output block
        x = self.output(x)
        x = x.transpose(2, 3)

        return {
            'prediction': x,
            'repr': repr
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.gso.train(mode)
        return self

    def eval(self):
        super().eval()
        self.gso.eval()
        return self


class STGCNWeakenedMP(nn.Module):
    """
    STGCN with Weakened Message Passing

    Interpolate between full message passing and no message passing:
    A' = alpha * A + (1 - alpha) * I

    Args:
        mp_strength: Message passing strength (0.0 = no MP, 1.0 = full MP)
    """

    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate,
                 mp_strength: float = 1.0):
        super(STGCNWeakenedMP, self).__init__()
        self.gso = gso
        self.T = T
        self.n_vertex = n_vertex
        self.mp_strength = mp_strength

        # Modify GSO to weaken message passing
        if mp_strength < 1.0:
            self._weaken_gso(gso, mp_strength, n_vertex)

        # ST-Conv blocks
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, bias, droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."

        self.output = OutputBlock(
            Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, bias, droprate)

    def _weaken_gso(self, gso, alpha, n_vertex):
        """Modify GSO to interpolate with identity matrix"""
        # GSO stores cheb_poly which is list of [I, G, 2GG-I, ...]
        # We modify G (index 1) to be alpha*G + (1-alpha)*I
        if hasattr(gso, 'cheb_poly') and len(gso.cheb_poly) > 1:
            device = gso.cheb_poly[1].device
            I = torch.eye(n_vertex, device=device)
            G_original = gso.cheb_poly[1].clone()
            G_weakened = alpha * G_original + (1 - alpha) * I
            gso.cheb_poly[1] = G_weakened

            # Recompute higher order polynomials
            if len(gso.cheb_poly) > 2:
                gso.cheb_poly[2] = 2 * torch.mm(G_weakened, gso.cheb_poly[1]) - I

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        x = history_data.permute(0, 3, 1, 2).contiguous()
        x = self.st_blocks(x)
        repr = x.clone()
        x = self.output(x)
        x = x.transpose(2, 3)

        return {
            'prediction': x,
            'repr': repr
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.gso.train(mode)
        return self

    def eval(self):
        super().eval()
        self.gso.eval()
        return self
