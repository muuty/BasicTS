import torch
import torch.nn as nn
from .layers import OutputBlock


# baselines/STGCN/arch/decoder.py
import torch
import torch.nn as nn
from .layers import OutputBlock


class STGCNDecoder(nn.Module):
    """
    STGCN Decoder: OutputBlock
    
    Central, Split, HierSplit 모두에서 사용
    """
    
    def __init__(
        self,
        Ko: int,
        last_block_channel: int,
        output_channels: list,
        out_steps: int,
        num_nodes: int,
        act_func: str,
        bias: bool,
        droprate: float,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.output = OutputBlock(
            Ko=Ko,
            last_block_channel=last_block_channel,
            channels=output_channels,
            end_channel=out_steps,
            n_vertex=num_nodes,
            act_func=act_func,
            bias=bias,
            droprate=droprate,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T', N, D)
        Returns:
            prediction: (B, out_steps, N, 1)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.output(x)
        x = x.permute(0, 1, 3, 2)

        return x