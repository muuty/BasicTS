import torch
import torch.nn as nn

from .staeformer_arch import STAEformer, SelfAttentionLayer


class GatedAttentionLayer(nn.Module):
    """AttentionLayer with learnable value gating.

    Same as AttentionLayer but adds a learned gate on Value vectors.
    The gate is computed from the raw input (before Q/K/V projection)
    so it can detect input patterns like sensor failure.

    Gate output is [0, 1] per node — when 0, that node's value contribution
    is completely suppressed regardless of attention weight.
    """

    def __init__(self, model_dim, num_heads=8, mask=False, gate_temperature=1.0, gate_init_bias=0.0, gate_hidden_dim=0):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.gate_temperature = gate_temperature

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        # Value gate: input embedding -> scalar gate per node
        if gate_hidden_dim > 0:
            self.gate_net = nn.Sequential(
                nn.Linear(model_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            if gate_init_bias != 0.0:
                nn.init.constant_(self.gate_net[-1].bias, gate_init_bias)
        else:
            self.gate_net = nn.Linear(model_dim, 1)
            if gate_init_bias != 0.0:
                nn.init.constant_(self.gate_net.bias, gate_init_bias)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]

        # Compute gate from raw value input (before projection)
        gate = torch.sigmoid(self.gate_net(value) * self.gate_temperature)  # (batch_size, ..., src_length, 1)

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Apply gate to value before splitting heads
        value = value * gate  # broadcast: [..., model_dim] * [..., 1]

        # Standard multi-head attention
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            tgt_length = query.shape[-2]
            src_length = key.shape[-1]
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class GatedSelfAttentionLayer(nn.Module):
    """SelfAttentionLayer using GatedAttentionLayer."""

    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False,
                 gate_temperature=1.0, gate_init_bias=0.0, gate_hidden_dim=0):
        super().__init__()

        self.attn = GatedAttentionLayer(model_dim, num_heads, mask, gate_temperature, gate_init_bias, gate_hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformerGated(STAEformer):
    """STAEformer with value-gated spatial attention.

    Only spatial attention layers are replaced with gated versions.
    Temporal attention remains standard (no gating needed there).
    """

    def __init__(self, **kwargs):
        gate_temperature = kwargs.pop('gate_temperature', 1.0)
        gate_init_bias = kwargs.pop('gate_init_bias', 0.0)
        gate_hidden_dim = kwargs.pop('gate_hidden_dim', 0)
        super().__init__(**kwargs)
        feed_forward_dim = kwargs.get('feed_forward_dim', 256)
        num_heads = kwargs.get('num_heads', 4)
        num_layers = kwargs.get('num_layers', 3)
        dropout = kwargs.get('dropout', 0.1)

        # Replace spatial attention with gated version
        self.attn_layers_s = nn.ModuleList([
            GatedSelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout,
                                    gate_temperature=gate_temperature, gate_init_bias=gate_init_bias,
                                    gate_hidden_dim=gate_hidden_dim)
            for _ in range(num_layers)
        ])
