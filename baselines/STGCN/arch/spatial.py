import torch
import torch.nn as nn

from .layers import GraphConvLayer
from .layers import STConvBlock


class STGCNServerSpatial(nn.Module):
    """
    STGCN SplitLearning용 Server Spatial.

    기존 STGCN split 구현은 서버가 Identity(`STGCNServerSpatial`)라 서버 파라미터가 0개가 됩니다.
    STAEformer처럼 서버에 학습 가능한 모듈을 두기 위해, 시간 차원을 바꾸지 않는
    GraphConvLayer(Residual 포함)를 서버에서 수행합니다.

    입력/출력 shape:
      - in : (B, T, N, D)
      - out: (B, T, N, D)
    """

    def __init__(
        self,
        graph_conv_type: str,
        feature_dim: int,
        Ks: int,
        adj_matrix: torch.Tensor,
        bias: bool,
        droprate: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self._gcn = GraphConvLayer(
            graph_conv_type=graph_conv_type,
            c_in=feature_dim,
            c_out=feature_dim,
            Ks=Ks,
            gso=adj_matrix,
            bias=bias,
        )
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=droprate)

    def forward(self, x: torch.Tensor, graph: torch.Tensor | None) -> torch.Tensor:
        # (B, T, N, D) -> (B, D, T, N)
        x_ = x.permute(0, 3, 1, 2)
        x_ = self._gcn(x_)
        x_ = self._relu(x_)
        x_ = self._dropout(x_)
        # (B, D, T, N) -> (B, T, N, D)
        return x_.permute(0, 2, 3, 1)


class STGCNServerSTBlock(nn.Module):
    """
    STGCN 서버용 ST-Conv 블록 (T-G-T-N-D 구조).

    목적:
    - "Transformer/attention 없이" STGCN의 구성요소를 유지하면서,
      split/hiersplit에서 서버 측에서 사용할 수 있는 블록을 제공.

    주의:
    - 원본 `STConvBlock`은 TemporalConvLayer(CausalConv2d)가 들어가 있어
      Kt>1이면 시간축 길이가 줄어듭니다.
    - split/hiersplit에서 시간축 길이를 유지하려면 보통 `preserve_time=True`로 두고
      내부적으로 effective_Kt=1로 동작시키는 것이 안전합니다.

    입력/출력:
      - in : (B, T, N, C)
      - out: (B, T', N, C)  (preserve_time=True면 T'=T)
    """

    def __init__(
        self,
        Kt: int,
        Ks: int,
        num_nodes: int,
        in_channels: int,
        mid_channels: int = 16,
        act_func: str = "glu",
        graph_conv_type: str = "cheb_graph_conv",
        adj_matrix: torch.Tensor | None = None,
        bias: bool = True,
        droprate: float = 0.0,
        preserve_time: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if adj_matrix is None:
            raise ValueError("STGCNServerSTBlock requires adj_matrix (gso).")

        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.mid_channels = int(mid_channels)
        self.preserve_time = bool(preserve_time)

        effective_Kt = 1 if self.preserve_time else int(Kt)

        # STConvBlock expects x: [B, C, T, N]
        # Keep feature dim: channels = [in, mid, in]
        self.block = STConvBlock(
            Kt=int(effective_Kt),
            Ks=int(Ks),
            n_vertex=self.num_nodes,
            last_block_channel=self.in_channels,
            channels=[self.in_channels, self.mid_channels, self.in_channels],
            act_func=str(act_func),
            graph_conv_type=str(graph_conv_type),
            gso=adj_matrix,
            bias=bool(bias),
            droprate=float(droprate),
        )

    def forward(self, x: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, N, C) -> (B, C, T, N)
        x_ = x.permute(0, 3, 1, 2)
        x_ = self.block(x_)
        # (B, C, T', N) -> (B, T', N, C)
        return x_.permute(0, 2, 3, 1)
