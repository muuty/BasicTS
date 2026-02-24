"""
ST-SSL Encoder: Spatio-Temporal Encoder with Chebyshev Graph Convolutions.

Reference: https://github.com/Echo-Ji/ST-SSL/blob/master/model/layers.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .augmentations import sim_global


class Align(nn.Module):
    """Align the input and output channel dimensions."""

    def __init__(self, c_in: int, c_out: int):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n, c, l, v)
        """
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConvLayer(nn.Module):
    """Temporal convolution layer with optional GLU activation."""

    def __init__(self, kt: int, c_in: int, c_out: int, act: str = "relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n, c, l, v)

        Returns:
            (n, c, l-kt+1, v)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(nn.Module):
    """Spatial convolution layer using Chebyshev polynomials."""

    def __init__(self, ks: int, c_in: int, c_out: int):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x: torch.Tensor, Lk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n, c, l, v)
            Lk: (K, v, v) Chebyshev polynomials

        Returns:
            (n, c, l, v)
        """
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        x_in = self.align(x)
        return torch.relu(x_gc + x_in)


class FCLayer(nn.Module):
    """Fully connected layer implemented as 1x1 convolution."""

    def __init__(self, c_in: int, c_out: int):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Pooler(nn.Module):
    """Pooling the token representations of region time series into the region level."""

    def __init__(self, n_query: int, d_model: int, agg: str = 'avg'):
        """
        Args:
            n_query: number of query (temporal length after first temporal conv)
            d_model: dimension of model
            agg: aggregation type, 'avg' or 'max'
        """
        super(Pooler, self).__init__()

        self.att = FCLayer(d_model, n_query)
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)

        self.d_model = d_model
        self.n_query = n_query
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: key sequence of region embedding, nclv

        Returns:
            x: hidden embedding used for conv, ncqv
            x_agg: region embedding for spatial similarity, nvc
            A: temporal attention, lnv
        """
        x_in = self.align(x)[:, :, -self.n_query:, :]  # ncqv

        # Calculate the attention matrix A using key x
        A = self.att(x)  # x: nclv, A: nqlv
        A = F.softmax(A, dim=2)  # nqlv

        # Calculate region embedding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2)  # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg)  # ncv->nvc

        # Calculate the temporal similarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2))  # A: lnqv->lnv

        return torch.relu(x + x_in), x_agg.detach(), A.detach()


class STEncoder(nn.Module):
    """
    Spatio-Temporal Encoder for ST-SSL.

    Uses two ST blocks with temporal and spatial convolutions,
    plus a pooler for computing similarity matrices.
    """

    def __init__(
            self,
            Kt: int = 3,
            Ks: int = 3,
            blocks: list = None,
            input_length: int = 12,
            num_nodes: int = 307,
            droprate: float = 0.1
    ):
        """
        Args:
            Kt: kernel size for temporal convolution
            Ks: kernel size for spatial convolution (Chebyshev order)
            blocks: channel configuration, e.g., [[2, 32, 64], [64, 32, 64]]
            input_length: length of input sequence
            num_nodes: number of nodes
            droprate: dropout rate
        """
        super(STEncoder, self).__init__()

        if blocks is None:
            blocks = [[2, 32, 64], [64, 32, 64]]

        self.Ks = Ks

        # First ST block
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.pooler = Pooler(input_length - (Kt - 1), c[1])

        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        # Second ST block
        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")

        self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)

        # Similarity matrices (populated during forward pass)
        self.s_sim_mx = None
        self.t_sim_mx = None

        # Output block
        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)

        self.receptive_field = input_length + Kt - 1

    def _cheb_polynomial(self, laplacian: torch.Tensor, K: int) -> torch.Tensor:
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        Args:
            laplacian: the graph laplacian, [v, v].

        Returns:
            the multi order Chebyshev laplacian, [K, v, v].
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    def _cal_laplacian(self, graph: torch.Tensor) -> torch.Tensor:
        """
        Return the normalized laplacian of the graph.

        Args:
            graph: the graph structure **without** self loop, [v, v].

        Returns:
            graph laplacian.
        """
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + I  # add self-loop to prevent zero in D
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        L = I - torch.mm(torch.mm(D, graph), D)
        return L

    def forward(self, x0: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: input tensor, [n, l, v, c]
            graph: adjacency matrix, [v, v]

        Returns:
            output tensor, [n, 1, v, c]
        """
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)

        in_len = x0.size(1)  # x0: nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0, 0, 0, 0, self.receptive_field - in_len, 0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes), nclv

        # ST block 1
        x = self.tconv11(x)  # nclv
        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        x = self.sconv12(x, Lk)  # nclv
        x = self.tconv13(x)
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        # ST block 2
        x = self.tconv21(x)
        x = self.sconv22(x, Lk)
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        # Output block
        x = self.out_conv(x)  # ncl(=1)v
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1)))  # nlvc

        return x  # nl(=1)vc


class MLP(nn.Module):
    """MLP predictor for traffic flow prediction."""

    def __init__(self, c_in: int, c_out: int):
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, [n, l, v, c]

        Returns:
            output tensor, [n, l, v, c_out]
        """
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2)))  # nlvc->nclv
        x = self.fc2(x).permute(0, 2, 3, 1)  # nclv->nlvc
        return x
