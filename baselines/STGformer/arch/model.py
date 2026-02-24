import torch
import torch.nn as nn
from typing import List

from .encoder import STGformerEncoder
from .spatial import STGformerSpatial
from .decoder import STGformerDecoder


class STGformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int,
        input_dim: int,
        output_dim: int,
        input_embedding_dim: int,
        tod_embedding_dim: int,
        dow_embedding_dim: int,
        spatial_embedding_dim: int,
        adaptive_embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        mlp_ratio: float = 2,
        use_mixed_proj: bool = True,
        dropout_a: float = 0.3,
        kernel_size: List[int] = [1],
        supports: List[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.encoder = STGformerEncoder(
            num_nodes=num_nodes,
            in_steps=in_steps,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            dropout_a=dropout_a,
            kernel_size=kernel_size,
            supports=supports,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.spatial = STGformerSpatial(
            model_dim=model_dim,
            mlp_ratio=mlp_ratio,
            num_nodes=num_nodes,
            in_steps=in_steps,
            adaptive_embedding_dim=adaptive_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            kernel_size=kernel_size,  # 리스트 전달
            supports=supports,
            order=2,
            qkv_bias=False,
        )


        self.decoder = STGformerDecoder(
            model_dim=model_dim,
            out_steps=out_steps,
            output_dim=output_dim,
            in_steps=in_steps,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=num_layers,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        x, graph = self.encoder(history_data)
        if graph is None:
            raise ValueError("Adaptive graph is required for STGformerModel.")
        x = self.spatial(x, graph)  # 한 번만 호출

        out = self.decoder(x)
        return out


































