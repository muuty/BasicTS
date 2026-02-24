import torch
import torch.nn as nn
from typing import Tuple, Optional
from .layers import STConvBlock


class STGCNEncoder(nn.Module):
    def __init__(
        self,
        Kt: int,
        Ks: int,
        num_nodes: int,
        last_block_channel: int,
        channels: list,
        channels_2: list,  # 두 번째 블록 채널
        act_func: str,
        graph_conv_type: str,
        adj_matrix: torch.Tensor,
        bias: bool,
        droprate: float,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # 첫 번째 STConvBlock
        self.st_block1 = STConvBlock(
            Kt=Kt,
            Ks=Ks,
            n_vertex=num_nodes,
            last_block_channel=last_block_channel,
            channels=channels,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=adj_matrix,
            bias=bias,
            droprate=droprate,
        )
        
        # 두 번째 STConvBlock (이거 있어야 함!)
        self.st_block2 = STConvBlock(
            Kt=Kt,
            Ks=Ks,
            n_vertex=num_nodes,
            last_block_channel=channels[-1],  # 첫 번째 출력 = 64
            channels=channels_2,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=adj_matrix,
            bias=bias,
            droprate=droprate,
        )
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.st_block1(x)
        x = self.st_block2(x)
        x = x.permute(0, 2, 3, 1)
        return x, None