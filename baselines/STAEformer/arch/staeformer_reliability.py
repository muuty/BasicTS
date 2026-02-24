"""STAEformer with External Reliability-Aware Attention Bias.

Like STAEformerCredibility, but reliability scores come from an external
encoder (DenoisingEncoder with reliability head) rather than being computed
internally. The encoder learns reliability during pre-training via
reliability-weighted reconstruction loss, then outputs per-node, per-timestep
reliability scores (0-1) as an extra input channel.

Reliability -> attention bias: log(r + eps) * learnable_scale_per_head
  - r=1 (reliable):   bias=0 (no effect)
  - r=0 (unreliable): bias=-large (suppressed in softmax)

Reuses CredibilitySelfAttentionLayer from staeformer_credibility.py.
"""

import torch
import torch.nn as nn

from baselines.layers import SelfAttentionLayer
from .staeformer_credibility import CredibilitySelfAttentionLayer


class STAEformerReliability(nn.Module):
    """STAEformer with external reliability-aware spatial attention.

    Expects reliability as the last channel of input:
      [physical_channels..., tod, dow, reliability]

    The reliability score (0-1) is converted to a pre-softmax attention bias
    via log(r + eps) with per-head learnable scale.
    """

    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        tod_index=-3,
        dow_index=-2,
        reliability_channel_idx=-1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.tod_index = tod_index
        self.dow_index = dow_index
        self.reliability_channel_idx = reliability_channel_idx
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # Input projection
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        # Output projection
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        # Temporal attention (standard, no reliability bias)
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Spatial attention (with reliability bias support)
        self.attn_layers_s = nn.ModuleList([
            CredibilitySelfAttentionLayer(
                self.model_dim, feed_forward_dim, num_heads, dropout
            )
            for _ in range(num_layers)
        ])

        # Per-head learnable scale for reliability bias
        self.reliability_scale = nn.Parameter(torch.ones(num_heads))

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data
        batch_size = x.shape[0]

        # Extract reliability from input (last channel)
        reliability = x[..., self.reliability_channel_idx]  # (B, T, N)

        # Extract temporal indices
        if self.tod_embedding_dim > 0:
            tod = x[..., self.tod_index] * self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., self.dow_index] * 7
        x = x[..., : self.input_dim]

        # Input projection + embeddings
        x = self.input_proj(x)  # (B, T, N, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            features.append(self.tod_embedding(tod.long()))
        if self.dow_embedding_dim > 0:
            features.append(self.dow_embedding(dow.long()))
        if self.spatial_embedding_dim > 0:
            features.append(
                self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
            )
        if self.adaptive_embedding_dim > 0:
            features.append(
                self.adaptive_embedding.expand(
                    size=(batch_size, *self.adaptive_embedding.shape)
                )
            )
        x = torch.cat(features, dim=-1)  # (B, T, N, model_dim)

        # Compute spatial attention bias from external reliability
        # log(r) maps: r=1 -> 0 (neutral), r->0 -> -inf (suppressed)
        eps = 1e-6
        log_r = torch.log(reliability + eps)  # (B, T, N)
        # Per-head scale: (num_heads,) -> (1, num_heads, 1, 1)
        scale = self.reliability_scale.view(1, self.num_heads, 1, 1)
        # Reshape for attention: (B, num_heads, T, N) -> (B*num_heads, T, 1, N)
        spatial_bias = (log_r.unsqueeze(1) * scale)  # (B, num_heads, T, N)
        num_nodes = reliability.shape[2]
        spatial_bias = spatial_bias.reshape(
            batch_size * self.num_heads, self.in_steps, 1, num_nodes
        )

        # Temporal attention (no reliability bias)
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        # Spatial attention (with reliability bias)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2, attn_bias=spatial_bias)

        # Output projection
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (B, N, T, D)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (B, out_steps, N, output_dim)
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))

        return {"prediction": out}
