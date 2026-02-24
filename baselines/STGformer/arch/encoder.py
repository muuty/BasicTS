import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class STGformerEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        steps_per_day: int,
        input_dim: int,
        input_embedding_dim: int,
        tod_embedding_dim: int,
        dow_embedding_dim: int,
        spatial_embedding_dim: int,
        adaptive_embedding_dim: int,
        dropout_a: float,
        kernel_size: List[int],
        supports: list,
        mlp_ratio: float,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.kernel_size = kernel_size[0]

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        else:
            self.tod_embedding = None
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        else:
            self.adaptive_embedding = None

        self.dropout = nn.Dropout(dropout_a)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.kernel_size), stride=1)
        self.temporal_proj = nn.Conv2d(
            model_dim, model_dim, (1, self.kernel_size), 1, 0
        )
        self.supports = supports

    def forward(self, history_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = history_data.shape[0]
        x = history_data
        if self.tod_embedding is not None:
            tod = x[..., -2] * self.steps_per_day
        else:
            tod = None
        if self.dow_embedding is not None:
            dow = x[..., -1] * 7
        else:
            dow = None
        x = x[..., : self.input_dim]
        x = self.input_proj(x)

        features = torch.tensor([], device=x.device, dtype=x.dtype)
        if tod is not None:
            tod_emb = self.tod_embedding(tod.long())
            features = torch.concat([features, tod_emb], -1)
        if dow is not None:
            dow_emb = self.dow_embedding(dow.long())
            features = torch.concat([features, dow_emb], -1)
        if self.adaptive_embedding is not None:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features = torch.concat([features, self.dropout(adp_emb)], -1)
        x = torch.cat([x, features], dim=-1)
        x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)

        graph = None
        if self.adaptive_embedding is not None:
            graph = torch.matmul(
                self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2)
            )
            graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
            graph = F.softmax(F.relu(graph), dim=-1)
        return x, graph


































