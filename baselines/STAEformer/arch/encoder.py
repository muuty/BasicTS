# encoder.py
import torch
import torch.nn as nn
from typing import Tuple, Optional

from baselines.layers import SelfAttentionLayer
from .pattern_bank_embedding import PatternBankEmbedding

class STAEformerEncoder(nn.Module):
    """
    STAEformer Encoder: Embedding + Temporal Attention

    원본 STAEformer의 앞부분:
    - Input projection
    - ToD/DoW/Spatial/Adaptive embeddings
    - Temporal Attention (attn_layers_t)
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        tod_index: int = -2,
        dow_index: int = -1,
        num_patterns: int = 0,
        node_mask_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.tod_index = tod_index
        self.dow_index = dow_index
        self.num_patterns = num_patterns
        self.node_mask_ratio = node_mask_ratio
        self.learnable_default = kwargs.get("learnable_default", False)

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        # Embeddings
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(num_nodes, spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            if num_patterns > 0:
                self.adaptive_embedding = PatternBankEmbedding(
                    num_nodes, in_steps, adaptive_embedding_dim, num_patterns)
            else:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
                )
            # Learnable default embedding for masked/new nodes
            if self.learnable_default and self.node_mask_ratio > 0:
                self.default_embedding = nn.Parameter(
                    torch.zeros(in_steps, 1, adaptive_embedding_dim)
                )

        # Temporal Attention layers
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, history_data: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            history_data: (B, T, N, C) where C includes input_dim + tod + dow

        Returns:
            temporal_features: (B, T, N, model_dim)
            graph: None (STAEformer doesn't use explicit graph)
        """
        x = history_data
        batch_size = x.shape[0]

        # Extract ToD/DoW information (indices configurable for different channel layouts)
        if self.tod_embedding_dim > 0:
            tod = x[..., self.tod_index] * self.steps_per_day

        if self.dow_embedding_dim > 0:
            dow = x[..., self.dow_index] * 7

        # Input projection (first input_dim channels = physical features)
        x = x[..., :self.input_dim]
        x = self.input_proj(x)  # (B, T, N, input_embedding_dim)

        # Collect features
        features = [x]

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(tod.long())
            features.append(tod_emb)

        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)

        if self.adaptive_embedding_dim > 0:
            if self.num_patterns > 0:
                adp_emb = self.adaptive_embedding()  # PatternBankEmbedding forward
            else:
                adp_emb = self.adaptive_embedding  # nn.Parameter
            adp_emb = adp_emb.expand(batch_size, *adp_emb.shape)
            # Masked Node Pre-training: randomly replace some nodes' embedding
            if self.training and self.node_mask_ratio > 0:
                node_mask = torch.rand(self.num_nodes, device=adp_emb.device) < self.node_mask_ratio
                if node_mask.any():
                    adp_emb = adp_emb.clone()
                    if self.learnable_default:
                        # Use learnable default embedding
                        fill = self.default_embedding.expand(
                            batch_size, -1, node_mask.sum(), -1
                        )
                    else:
                        # Use mean of current embeddings
                        fill = adp_emb.mean(dim=2, keepdim=True).expand(
                            -1, -1, node_mask.sum(), -1
                        )
                    adp_emb[:, :, node_mask, :] = fill
            features.append(adp_emb)

        # Concatenate all features
        x = torch.cat(features, dim=-1)  # (B, T, N, model_dim)

        # Temporal Attention
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        return x, None  # No graph for STAEformer
