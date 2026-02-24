"""
STAEformer with Temporal Encoder (End-to-End)

Key idea from design.md:
- Add TemporalEncoder BEFORE STAEformer
- Train End-to-End (no pre-training, no frozen encoder)
- TemporalEncoder learns temporal representations
- STAEformer receives encoded features instead of raw input

Architecture:
    Input [B, T, N, C]
        → TemporalEncoder [B, T, N, D]
        → STAEformer [B, T', N, 1]
"""
import torch
import torch.nn as nn

from .staeformer_arch import STAEformer, SelfAttentionLayer


class TemporalEncoder(nn.Module):
    """
    Transformer-based temporal encoder.

    Processes each node independently, learning temporal patterns.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(c_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            z: [B, T, N, D] - encoded representations
        """
        B, T, N, C = x.shape

        # Reshape: treat each node independently
        # [B, T, N, C] -> [B*N, T, C]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # Project to d_model
        x = self.input_proj(x)  # [B*N, T, D]

        # Encode with transformer (no causal mask for bidirectional)
        x = self.transformer(x)  # [B*N, T, D]

        # Reshape back: [B*N, T, D] -> [B, T, N, D]
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        return x


class STAEformerWithTemporalEncoder(nn.Module):
    """
    STAEformer with Temporal Encoder (End-to-End Training)

    Architecture:
        1. TemporalEncoder: learns temporal representations [B, T, N, C] -> [B, T, N, D]
        2. Modified STAEformer: spatiotemporal attention on encoded features

    Key benefits:
        - Joint optimization of encoder and forecaster
        - No need for separate pre-training
        - Temporal encoder learns task-specific representations
    """

    def __init__(
        self,
        # TemporalEncoder params
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        input_dim: int = 3,  # raw input: speed, tod, dow
        encoder_dim: int = 64,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        encoder_dropout: float = 0.1,
        # STAEformer-like params
        steps_per_day: int = 288,
        output_dim: int = 1,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
        **kwargs
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.encoder_dim = encoder_dim
        self.steps_per_day = steps_per_day
        self.use_mixed_proj = use_mixed_proj

        # Temporal Encoder (trainable)
        self.temporal_encoder = TemporalEncoder(
            c_in=1,  # Only encode the main feature (speed)
            d_model=encoder_dim,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            dropout=encoder_dropout,
        )

        # Embedding dimensions
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        # Model dimension = encoder_dim + embeddings
        self.model_dim = encoder_dim + tod_embedding_dim + dow_embedding_dim + adaptive_embedding_dim

        # Embeddings (similar to STAEformer)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        # STAEformer-style attention layers
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> dict:
        """
        Args:
            history_data: [B, T, N, C] where C = [speed, tod, dow]

        Returns:
            dict with 'prediction': [B, T', N, 1]
        """
        batch_size = history_data.shape[0]

        # Extract features
        main_feature = history_data[..., 0:1]  # [B, T, N, 1] - speed
        tod = history_data[..., 1]  # [B, T, N]
        dow = history_data[..., 2]  # [B, T, N]

        # Encode main feature with temporal encoder
        encoded = self.temporal_encoder(main_feature)  # [B, T, N, D]

        # Build feature list
        features = [encoded]

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding((dow * 7).long())
            features.append(dow_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape)
            features.append(adp_emb)

        # Concatenate all features
        x = torch.cat(features, dim=-1)  # [B, T, N, model_dim]

        # Temporal attention
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        # Spatial attention
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        # Output projection
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # [B, N, T, model_dim]
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)  # [B, T', N, output_dim]
        else:
            out = x.transpose(1, 3)  # [B, model_dim, N, T]
            out = self.temporal_proj(out)  # [B, model_dim, N, T']
            out = self.output_proj(out.transpose(1, 3))  # [B, T', N, output_dim]

        return {'prediction': out}
