import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, kernel: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(
            2 * model_dim if kernel != 12 else model_dim, model_dim
        )
        self.fast = 1

    def forward(self, x: torch.Tensor, edge_index=None, dim: int = 0) -> torch.Tensor:
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        if self.fast:
            out_s = self.fast_attention(x, qs, ks, vs, dim=dim)
        else:
            out_s = self.normal_attention(x, qs, ks, vs, dim=dim)
        if x.size(1) > 1:
            qs = torch.stack(
                torch.split(query.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            ks = torch.stack(
                torch.split(key.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            vs = torch.stack(
                torch.split(value.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            if self.fast:
                out_t = self.fast_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            else:
                out_t = self.normal_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            out = torch.concat([out_s, out_t], -1)
            out = self.out_proj(out)
        else:
            out = self.out_proj(out_s)
        return out

    def fast_attention(
        self, x: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, dim: int
    ) -> torch.Tensor:
        qs = nn.functional.normalize(qs, dim=-1)
        ks = nn.functional.normalize(ks, dim=-1)
        N = qs.shape[1]
        b, l = x.shape[dim : dim + 2]

        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)
        attention_num += N * vs

        all_ones = torch.ones([ks.shape[1]], device=ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        out = attention_num / attention_normalizer
        out = torch.unflatten(out, dim, (b, l)).flatten(start_dim=3)
        return out

    def normal_attention(
        self, x: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, dim: int
    ) -> torch.Tensor:
        b, l = x.shape[dim : dim + 2]
        qs, ks, vs = qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = torch.unflatten(x, dim, (b, l)).flatten(start_dim=3)
        return x


class GraphPropagate(nn.Module):
    def __init__(self, Ks: int, gso: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.Ks = Ks
        self.gso = gso
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> List[torch.Tensor]:
        if self.Ks < 1:
            raise ValueError(f"ERROR: Ks must be a positive integer, received {self.Ks}.")
        x_k = x
        x_list: List[torch.Tensor] = [x]
        for k in range(1, self.Ks):
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))
        return x_list


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_ratio: float,
        num_heads: int,
        dropout: float,
        kernel: int,
        supports: list,
        order: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.locals = GraphPropagate(Ks=order, gso=supports[0], dropout=dropout)
        self.kernel_size = kernel
        self.attn = nn.ModuleList(
            [
                FastAttentionLayer(model_dim, num_heads, kernel=kernel, qkv_bias=qkv_bias)
                for _ in range(order)
            ]
        )
        self.pws = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(order)])
        for proj in self.pws:
            nn.init.constant_(proj.weight, 0)
            nn.init.constant_(proj.bias, 0)

        self.fc = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = [1, 0.01, 0.001]

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        x_loc = self.locals(x, graph)
        c = x
        x_glo = x  # 참조 유지 (clone 불필요)
        
        for i, z in enumerate(x_loc):
            att_outputs = self.attn[i](z)
            x_glo = x_glo + att_outputs * self.pws[i](c) * self.scale[i]
            c = att_outputs
        
        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x

class STGformerSpatial(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_ratio: float,
        num_nodes: int,
        in_steps: int,
        num_heads: int,
        dropout: float,
        supports: list,
        order: int = 2,
        qkv_bias: bool = False,
        kernel_size: list[int] = [1],
        adaptive_embedding_dim: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, list) else kernel_size
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.kernel_size), stride=1)
        self.adaptive_embedding = nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
        nn.init.xavier_uniform_(self.adaptive_embedding)    
        self.attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(
                    model_dim=model_dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    dropout=dropout,
                    kernel=size,
                    supports=supports,
                    order=order,
                    qkv_bias=qkv_bias,
                )
                for size in kernel_size
            ]
        )

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
        graph = graph.transpose(0, 2)            # (N, N, T)
        graph = self.pooling(graph.unsqueeze(1))  # (N, 1, N, T) -> pooled
        graph = graph.squeeze(1).transpose(0, 2)  # (T, N, N)
        graph = F.softmax(F.relu(graph), dim=-1)
        
        for attn in self.attn_layers:
            x = attn(x, graph)
        return x

