# model.py
import torch
import torch.nn as nn
from typing import List, Optional

from .encoder import STAEformerEncoder
from .spatial import STAEformerSpatial
from .decoder import STAEformerDecoder


class STAEformer(nn.Module):
    """
    STAEformer: Spatio-Temporal Adaptive Embedding Transformer
    
    Paper: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
    Link: https://arxiv.org/abs/2308.10425
    Venue: CIKM 2023
    
    Structure:
        Encoder: Embedding + Temporal Attention
        Spatial: Spatial Attention
        Decoder: Output Projection
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
        tod_index: int = -2,
        dow_index: int = -1,
        num_patterns: int = 0,
        **kwargs,
    ):
        super().__init__()

        model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.encoder = STAEformerEncoder(
            num_nodes=num_nodes,
            in_steps=in_steps,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            tod_index=tod_index,
            dow_index=dow_index,
            num_patterns=num_patterns,
            node_mask_ratio=kwargs.get("node_mask_ratio", 0.0),
            learnable_default=kwargs.get("learnable_default", False),
        )
        
        self.spatial = STAEformerSpatial(
            model_dim=model_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.decoder = STAEformerDecoder(
            num_nodes=num_nodes,
            model_dim=model_dim,
            in_steps=in_steps,
            out_steps=out_steps,
            output_dim=output_dim,
            use_mixed_proj=use_mixed_proj,
        )
    
    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> dict:
        """
        Args:
            history_data: (B, in_steps, num_nodes, input_dim+tod+dow)
            future_data: Not used
            batch_seen: Current batch index
            epoch: Current epoch
            train: Training mode flag

        Returns:
            dict with 'prediction': (B, out_steps, num_nodes, output_dim)
        """
        # Encoder: Embedding + Temporal Attention
        x, graph = self.encoder(history_data)

        # Spatial: Spatial Attention
        x = self.spatial(x, graph)

        # Decoder: Output Projection
        out = self.decoder(x)

        return {'prediction': out}