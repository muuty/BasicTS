"""
GPT-ST Module Components.
Includes: MLP_RL, cap_adj, cap, hyperTem, hyperSpa, time_feature, squash function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Squash activation for capsule networks.

    Args:
        x: Input tensor
        dim: Dimension along which to compute the norm

    Returns:
        Squashed tensor with values between 0 and 1 in magnitude
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class MLP_RL(nn.Module):
    """MLP with learnable weights for generating classification logits.

    Used for adaptive masking in the encoder.
    """

    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int, embed_dim: int, device: str):
        super(MLP_RL, self).__init__()

        self.ln1 = nn.Linear(dim_in, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, dim_out)

        # Spatial weight pool
        self.weights_pool_spa = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_spa = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))

        # Temporal weight pool
        self.weights_pool_tem = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_tem = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))

        self.act = nn.LeakyReLU()
        self.device = device

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weights_pool_spa)
        nn.init.xavier_uniform_(self.weights_pool_tem)
        nn.init.zeros_(self.bias_pool_spa)
        nn.init.zeros_(self.bias_pool_tem)

    def forward(self, eb: torch.Tensor, time_eb: torch.Tensor, node_eb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eb: Input embedding [B, T, N, D_in]
            time_eb: Time embedding [B, T, embed_dim]
            node_eb: Node embedding [N, embed_dim]

        Returns:
            logits: Classification logits [B, T, N, D_out]
        """
        eb_out = self.ln1(eb)

        # Spatial transformation
        weights_spa = torch.einsum('nd,dio->nio', node_eb, self.weights_pool_spa)
        bias_spa = torch.matmul(node_eb, self.bias_pool_spa)
        out_spa = torch.einsum('btni,nio->btno', eb_out, weights_spa) + bias_spa
        out_spa = self.act(out_spa)

        # Temporal transformation
        weights_tem = torch.einsum('btd,dio->btio', time_eb, self.weights_pool_tem)
        bias_tem = torch.matmul(time_eb, self.bias_pool_tem).unsqueeze(-2)
        out_tem = torch.einsum('btni,btio->btno', out_spa, weights_tem) + bias_tem
        out_tem = self.act(out_tem)

        logits = self.ln3(out_tem)
        return logits


class TimeFeature(nn.Module):
    """Time feature encoder for day and week embeddings."""

    def __init__(self, embed_dim: int):
        super(TimeFeature, self).__init__()

        self.ln_day = nn.Linear(1, embed_dim)
        self.ln_week = nn.Linear(1, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eb: Time features [B, T, 2] where [..., 0] is day_of_day, [..., 1] is day_of_week

        Returns:
            Time embedding [B, T, embed_dim]
        """
        day = self.ln_day(eb[:, :, 0:1])
        week = self.ln_week(eb[:, :, 1:2])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb


class TimeFeatureSPG(nn.Module):
    """Time feature encoder for spatial-temporal graph (with larger input dimension)."""

    def __init__(self, embed_dim: int, input_dim: int = 12):
        super(TimeFeatureSPG, self).__init__()

        self.ln_day = nn.Linear(input_dim, embed_dim)
        self.ln_week = nn.Linear(input_dim, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eb: Time features [B, T, 2, input_dim]

        Returns:
            Time embedding [B, T, embed_dim]
        """
        day = self.ln_day(eb[:, :, 0])
        week = self.ln_week(eb[:, :, 1])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb


class HyperTem(nn.Module):
    """Temporal hypergraph convolution layer."""

    def __init__(self, timesteps: int, num_nodes: int, dim_in: int, dim_out: int,
                 embed_dim: int, HT_Tem: int):
        super(HyperTem, self).__init__()

        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HT_Tem, timesteps), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, eb: torch.Tensor, node_embeddings: torch.Tensor,
                time_eb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eb: Input embedding [B, T, N, D_in]
            node_embeddings: Node embeddings [N, embed_dim]
            time_eb: Time embeddings [B, T, embed_dim]

        Returns:
            Output embedding [B, T, N, D_out]
        """
        # Dynamic adjacency based on node embeddings
        adj_dynamics = torch.einsum('nk,kht->nht', node_embeddings, self.adj).permute(1, 2, 0)

        # Hypergraph message passing
        hyperEmbeds = torch.einsum('htn,btnd->bhnd', adj_dynamics, eb)
        retEmbeds = torch.einsum('thn,bhnd->btnd', adj_dynamics.transpose(0, 1), hyperEmbeds)

        # Dynamic weights based on time embeddings
        weights = torch.einsum('btd,dio->btio', time_eb, self.weights_pool)
        bias = torch.matmul(time_eb, self.bias_pool).unsqueeze(2)
        out = torch.einsum('btni,btio->btno', retEmbeds, weights) + bias

        return self.act(out + eb)


class HyperSpa(nn.Module):
    """Spatial hypergraph convolution layer."""

    def __init__(self, num_nodes: int, dim_in: int, dim_out: int,
                 embed_dim: int, HS_Spa: int):
        super(HyperSpa, self).__init__()

        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HS_Spa, num_nodes), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, eb: torch.Tensor, node_embeddings: torch.Tensor,
                time_eb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eb: Input embedding [B, T, N, D_in]
            node_embeddings: Node embeddings [N, embed_dim]
            time_eb: Time embeddings [B, T, embed_dim]

        Returns:
            Output embedding [B, T, N, D_out]
        """
        # Dynamic adjacency based on time embeddings
        adj_dynamics = torch.einsum('btk,khn->bthn', time_eb, self.adj).permute(1, 2, 0)

        # Hypergraph message passing
        hyperEmbeds = self.act(torch.einsum('bthn,btnd->bthd', adj_dynamics, eb))
        retEmbeds = self.act(torch.einsum('btnh,bthd->btnd', adj_dynamics.transpose(-1, -2), hyperEmbeds))

        # Dynamic weights based on node embeddings
        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        out = torch.einsum('btni,nio->btno', retEmbeds, weights) + bias

        return self.act(out + eb)


class Cap(nn.Module):
    """Capsule routing layer for spatial hypergraph learning."""

    def __init__(self, dim: int, num_nodes: int, timesteps: int, embed_dim: int,
                 embed_dim_spa: int, HS: int, HT: int, num_route: int):
        super(Cap, self).__init__()

        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.dim = dim
        self.num_route = num_route
        self.HS = HS
        self.TT = HS * timesteps

        self.ln_p = nn.Linear(dim, dim)
        self.t_adj = nn.Parameter(torch.randn(embed_dim_spa, HT, self.TT), requires_grad=True)
        self.adj = nn.Parameter(torch.randn(embed_dim_spa, HS, num_nodes), requires_grad=True)
        self.weights_spa = nn.Parameter(torch.FloatTensor(embed_dim, dim, dim))
        self.bias_spa = nn.Parameter(torch.FloatTensor(embed_dim, dim))

        self.LRelu = nn.LeakyReLU()

        # Temporal mask template
        mask_template = (torch.linspace(1, timesteps, steps=timesteps)) / 12.
        self.register_buffer('mask_template', mask_template)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.t_adj)
        nn.init.xavier_uniform_(self.adj)
        nn.init.xavier_uniform_(self.weights_spa)
        nn.init.zeros_(self.bias_spa)

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor,
                time_eb: torch.Tensor, teb: torch.Tensor) -> tuple:
        """
        Args:
            x: Input [B, T, N, D]
            node_embeddings: Node embeddings [N, embed_dim]
            time_eb: Time embeddings for SPG [B, T, embed_dim_spa]
            teb: Time embeddings [B, T, embed_dim_spa]

        Returns:
            out: Output [B, T, N, D]
            c: Routing coefficients (detached)
            dynamic_adj: Dynamic adjacency (detached)
        """
        batch_size = x.size(0)
        device = x.device

        Pcaps = self.ln_p(x)
        Pcaps_out = squash(Pcaps, dim=-1)

        # Dynamic adjacency
        dadj = torch.einsum('btd,dhn->bthn', teb, self.adj)
        test1 = torch.einsum('bthn,btnd->bthd', dadj.softmax(-2), Pcaps_out)

        Dcaps_in = torch.matmul(
            squash(test1).unsqueeze(-1).permute(0, 1, 3, 2, 4),
            Pcaps_out.unsqueeze(-1).permute(0, 1, 3, 2, 4).transpose(-1, -2)
        ).permute(0, 1, 3, 4, 2)

        k_test = Pcaps_out.detach()
        temp_u_hat = Dcaps_in.detach()

        # Capsule routing
        b = torch.zeros(batch_size, self.timesteps, self.HS, self.num_nodes, 1).to(device)
        for route_iter in range(self.num_route):
            c = b.softmax(dim=2)
            s = (c * temp_u_hat).sum(-2)
            v = squash(s)
            uv = torch.matmul(v, k_test.transpose(-1, -2)).unsqueeze(-1)
            b = b + uv

        c = (b + dadj.unsqueeze(-1)).softmax(dim=2)

        # Aggregate using routing coefficients
        s = torch.einsum('bthn,btnd->bthd', c.squeeze(-1), Pcaps_out)

        # Add temporal position encoding
        time_index = self.mask_template.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        hyperEmbeds_spa = s + time_index
        hyperEmbeds_spa = hyperEmbeds_spa.reshape(batch_size, -1, self.dim)

        # Temporal hypergraph transformation
        dynamic_adj = torch.einsum('bd,dhk->bhk', time_eb, self.t_adj)
        hyperEmbeds_tem = self.LRelu(torch.einsum('bhk,bkd->bhd', dynamic_adj, hyperEmbeds_spa))
        retEmbeds_tem = self.LRelu(torch.einsum('bkh,bhd->bkd', dynamic_adj.transpose(-1, -2), hyperEmbeds_tem))
        retEmbeds_tem = retEmbeds_tem.reshape(batch_size, self.timesteps, -1, self.dim) + s

        v = squash(retEmbeds_tem)
        reconstruction = torch.einsum('btnh,bthd->btnd', c.squeeze(-1).transpose(-1, -2), v)

        # Apply spatial weights
        weights_spatial = torch.einsum('nd,dio->nio', node_embeddings, self.weights_spa)
        bias_spatial = torch.matmul(node_embeddings, self.bias_spa)
        out = torch.einsum('btni,nio->btno', reconstruction, weights_spatial) + bias_spatial

        return self.LRelu(out + x), c.detach(), dynamic_adj.detach()


class CapAdj(nn.Module):
    """Capsule routing layer for adjacency learning (used in masking)."""

    def __init__(self, dim: int, num_nodes: int, timesteps: int, embed_dim: int,
                 embed_dim_spa: int, mask_R: float, HS: int, HT: int, num_route: int):
        super(CapAdj, self).__init__()

        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.dim = dim
        self.mask_R = mask_R
        self.num_route = num_route
        self.HS = HS
        self.TT = HS * timesteps

        self.ln_p = nn.Linear(dim, dim)
        self.adj = nn.Parameter(torch.randn(embed_dim_spa, HS, num_nodes), requires_grad=True)
        self.LRelu = nn.LeakyReLU()

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.adj)

    def forward(self, x: torch.Tensor, teb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, N, D]
            teb: Time embeddings [B, T, embed_dim_spa]

        Returns:
            c: Routing coefficients [B, T, HS, N, 1]
        """
        batch_size = x.size(0)
        device = x.device

        Pcaps = self.ln_p(x)
        Pcaps_out = squash(Pcaps, dim=-1)

        dadj = torch.einsum('btd,dhn->bthn', teb, self.adj)
        test1 = torch.einsum('bthn,btnd->bthd', dadj.softmax(-2), Pcaps_out)

        Dcaps_in = torch.matmul(
            squash(test1).unsqueeze(-1).permute(0, 1, 3, 2, 4),
            Pcaps_out.unsqueeze(-1).permute(0, 1, 3, 2, 4).transpose(-1, -2)
        ).permute(0, 1, 3, 4, 2)

        k_test = Pcaps_out.detach()
        temp_u_hat = Dcaps_in.detach()

        # Capsule routing
        b = torch.zeros(batch_size, self.timesteps, self.HS, self.num_nodes, 1).to(device)
        for route_iter in range(self.num_route):
            c = b.softmax(dim=2)
            s = (c * temp_u_hat).sum(-2)
            v = squash(s)
            uv = torch.matmul(v, k_test.transpose(-1, -2)).unsqueeze(-1)
            b = b + uv

        c = (b + dadj.unsqueeze(-1)).softmax(dim=2)
        return c


class Fusion(nn.Module):
    """Fusion layer to combine pretrained and input embeddings."""

    def __init__(self, dim: int):
        super(Fusion, self).__init__()
        self.HS_fc = nn.Linear(dim, dim, bias=True)
        self.HT_fc = nn.Linear(dim, dim, bias=True)
        self.output_fc = nn.Linear(dim, dim, bias=True)

    def forward(self, flow_eb: torch.Tensor, time_eb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow_eb: Flow embeddings from pretrained model [B, T, N, D]
            time_eb: Time-projected input embeddings [B, T, N, D]

        Returns:
            Fused embeddings [B, T, N, D]
        """
        XS = self.HS_fc(flow_eb)
        XT = self.HT_fc(time_eb)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, flow_eb), torch.multiply(1 - z, time_eb))
        H = self.output_fc(H)
        return H
