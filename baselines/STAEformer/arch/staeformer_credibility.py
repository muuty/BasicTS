"""STAEformer with Credibility-Aware Attention Bias.

Addresses attention collapse: dead/failing sensors get disproportionate
attention due to consistent key representations (softmax bias toward
constant inputs). Adds a learnable pre-softmax bias computed from raw
input embeddings to penalize uninformative nodes in spatial attention.

See docs/attention_collapse_missing_values.md for full analysis.
"""

import torch
import torch.nn as nn

from baselines.layers import SelfAttentionLayer


class CredibilityAttentionLayer(nn.Module):
    """Attention with optional pre-softmax credibility bias on key side."""

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.last_attn = None

    def forward(self, query, key, value, attn_bias=None):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        # Split into heads: (num_heads*B, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim**0.5
        # Apply credibility bias before softmax
        if attn_bias is not None:
            attn_score = attn_score + attn_bias
        if self.mask:
            mask = torch.ones(tgt_length, src_length, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)
        attn_score = torch.softmax(attn_score, dim=-1)
        self.last_attn = attn_score
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class CredibilitySelfAttentionLayer(nn.Module):
    """Self-attention that passes credibility bias to attention computation."""

    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()
        self.attn = CredibilityAttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2, attn_bias=None):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x, attn_bias=attn_bias)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class STAEformerCredibility(nn.Module):
    """STAEformer with credibility-aware spatial attention.

    Computes per-node, per-timestep credibility scores from raw embeddings
    (before temporal attention) and applies them as pre-softmax key-side bias
    in spatial attention. This directly counteracts the softmax bias toward
    dead/failing sensors with consistent key representations.

    Key design choices:
    - Bias computed from raw embedding (not post-temporal) to preserve flow signal
    - Key-side bias: controls how much each node is attended TO
    - Per-head output: each head can learn different credibility criteria
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
        tod_index=-2,
        dow_index=-1,
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

        # Temporal attention (standard, no credibility bias)
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Spatial attention (with credibility bias support)
        self.attn_layers_s = nn.ModuleList([
            CredibilitySelfAttentionLayer(
                self.model_dim, feed_forward_dim, num_heads, dropout
            )
            for _ in range(num_layers)
        ])

        # Credibility network: raw embedding -> per-head bias score
        # Negative output = "don't attend to this node", positive = "attend more"
        self.credibility_net = nn.Linear(self.model_dim, num_heads)

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data
        batch_size = x.shape[0]

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

        # Compute credibility from raw embedding (before temporal attention)
        # cred: (B, T, N, num_heads) - per node, per timestep, per head
        cred = self.credibility_net(x)
        # Reshape for spatial attention key-side bias: (num_heads*B, T, 1, N)
        num_nodes = x.shape[2]
        spatial_bias = cred.permute(0, 3, 1, 2).reshape(
            batch_size * self.num_heads, self.in_steps, 1, num_nodes
        )

        # Temporal attention (no credibility bias)
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        # Spatial attention (with credibility bias)
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
