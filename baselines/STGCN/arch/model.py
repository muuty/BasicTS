import torch
import torch.nn as nn

from typing import List
from .encoder import STGCNEncoder
from .decoder import STGCNDecoder


class STGCN(nn.Module):
    """
    Central Learning용 STGCN
    
    공통 컴포넌트(STGCNEncoder, STGCNDecoder)를 조합하여 구성
    """
    
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
    ) -> None:
        super().__init__()
        
        # Validation
        if len(blocks) < 5:
            raise ValueError("blocks는 최소 5개 필요: [in], [stconv1], [stconv2], [fc], [out]")
        
        # 시간 차원 계산: 2개 STConvBlock, 각각 2*(Kt-1) 감소
        num_st_blocks = 2
        Ko = T - num_st_blocks * 2 * (Kt - 1)
        if Ko <= 0:
            raise ValueError(f"Ko={Ko} must be positive. T={T}, Kt={Kt}")
        
        # Encoder: 2개의 STConvBlock
        self.encoder = STGCNEncoder(
            Kt=Kt,
            Ks=Ks,
            num_nodes=num_nodes,
            last_block_channel=blocks[0][0],  # input channel
            channels=blocks[1],  # 첫 번째 STConvBlock
            channels_2=blocks[2],  # 두 번째 STConvBlock
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            adj_matrix=adj_matrix,
            bias=bias,
            droprate=droprate,
        )
        
        # Decoder: OutputBlock
        self.decoder = STGCNDecoder(
            Ko=Ko,
            last_block_channel=blocks[2][-1],  # Encoder 출력 채널
            output_channels=blocks[-2],  # FC layers
            out_steps=blocks[-1][0],  # 출력 시간 길이
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
        """
        Args:
            history_data: (B, T, N, C)
        Returns:
            prediction: (B, out_steps, N, 1)
        """
        # Encoder
        x, _ = self.encoder(history_data)  # (B, T', N, D)
        
        # Decoder
        x = self.decoder(x)  # (B, out_steps, N, 1)
        
        return x