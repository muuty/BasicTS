"""
STMAE: Spatio-Temporal Masked Autoencoder.

A backbone-agnostic masked autoencoder framework for spatio-temporal data.
Supports any encoder backbone (AGCRN, GWNet, DCRNN, etc.) and adds:
1. Feature masking (temporal patches)
2. Structure masking (graph edges via random walks)
3. Structure decoder (inner product)
4. Feature decoder (reconstruction)

The framework supports two-stage training:
1. Pre-training: Self-supervised with masking and reconstruction losses
2. Fine-tuning: Supervised forecasting with frozen or fine-tuned encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from typing import Dict, Optional, Tuple, Any, Union

from .masking import FeatureMasking, StructureMasking
from .decoders import InnerProductDecoder, FeatureDecoder, ForecastingDecoder


# Backbone registry for string-based lookup (avoids pickle issues)
BACKBONE_REGISTRY = {}


def register_backbone(name: str):
    """Decorator to register a backbone class."""
    def decorator(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def get_backbone_class(backbone: Union[str, type]) -> type:
    """
    Get backbone class from string name or return class directly.

    Args:
        backbone: Either a string name (e.g., "AGCRN") or the class itself

    Returns:
        The backbone class
    """
    if backbone is None:
        return None
    if isinstance(backbone, str):
        # Try registry first
        if backbone in BACKBONE_REGISTRY:
            return BACKBONE_REGISTRY[backbone]
        # Try dynamic import for known backbones
        if backbone == "AGCRN":
            from baselines.AGCRN.arch import AGCRN
            return AGCRN
        elif backbone == "GWNet":
            from baselines.GWNet.arch import GWNet
            return GWNet
        elif backbone == "DCRNN":
            from baselines.DCRNN.arch import DCRNN
            return DCRNN
        else:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Available: {list(BACKBONE_REGISTRY.keys())}")
    # Already a class
    return backbone


class STMAE(nn.Module):
    """
    Spatio-Temporal Masked Autoencoder.

    Backbone-agnostic wrapper that adds masking strategies and decoders
    around any spatio-temporal encoder.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        input_len: int,
        output_len: int,
        # Backbone configuration
        backbone_class: type = None,
        backbone_params: Dict = None,
        # Masking configuration
        mask_f_ratio: float = 0.5,
        mask_s_ratio: float = 0.3,
        patch_length: int = 1,
        walks_per_node: int = 10,
        walk_length: int = 20,
        # Decoder configuration
        stru_dec_dropout: float = 0.0,
        stru_dec_proj: bool = False,
        # Loss weights
        sl_weight: float = 1.0,
        fl_weight: float = 1.0,
        # Node embeddings
        embed_dim: int = 10,
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension of the encoder
            input_len: Input sequence length
            output_len: Output/prediction sequence length
            backbone_class: Encoder backbone class
            backbone_params: Parameters for backbone instantiation
            mask_f_ratio: Feature masking ratio (0-1)
            mask_s_ratio: Structure masking ratio (0-1)
            patch_length: Patch length for feature masking
            walks_per_node: Random walks per node for structure masking
            walk_length: Walk length for structure masking
            stru_dec_dropout: Dropout for structure decoder
            stru_dec_proj: Whether to use projection in structure decoder
            sl_weight: Structure loss weight
            fl_weight: Feature loss weight
            embed_dim: Node embedding dimension
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        self.output_len = output_len
        self.mask_f_ratio = mask_f_ratio
        self.mask_s_ratio = mask_s_ratio
        self.patch_length = patch_length
        self.sl_weight = sl_weight
        self.fl_weight = fl_weight

        # Node embeddings for generating dynamic adjacency
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

        # Feature embedding layer
        self.to_feat_embedding = nn.Linear(input_dim, hidden_dim)

        # Masking modules
        self.feature_masking = FeatureMasking(patch_length=patch_length)
        self.structure_masking = StructureMasking(
            walks_per_node=walks_per_node,
            walk_length=walk_length,
        )

        # Build backbone encoder (resolve string to class if needed)
        resolved_backbone = get_backbone_class(backbone_class)
        if resolved_backbone is not None:
            self.encoder = resolved_backbone(**backbone_params)
        else:
            # Default: simple GRU-based encoder
            self.encoder = DefaultEncoder(
                num_nodes=num_nodes,
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )

        # Decoders
        self.structure_decoder = InnerProductDecoder(
            dropout=stru_dec_dropout,
            activation='none',  # Raw logits for BCE loss
            with_proj=stru_dec_proj,
            hidden_dim=hidden_dim,
        )
        self.feature_decoder = FeatureDecoder(
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_length=input_len,
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters following STGCL."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def get_support(self) -> torch.Tensor:
        """
        Get current learned support/adjacency matrix.

        Returns:
            support: [N, N] adjacency matrix with values in (0, 1)
        """
        support = torch.sigmoid(
            torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))
        )
        return support

    def encode(
        self,
        x: torch.Tensor,
        mask_s: float = 0,
        mask_f: float = 0,
        f_mask: torch.Tensor = None,
        s_mask: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input with optional masking.

        Args:
            x: Input tensor [B, T, N, D]
            mask_s: Structure masking ratio (0 = no masking)
            mask_f: Feature masking ratio (0 = no masking)
            f_mask: Pre-computed feature mask [B, T, N]
            s_mask: Pre-computed structure mask [N, N]

        Returns:
            embedding: Full temporal embedding [B, T, N, H]
            summary: Summary embedding (last timestep) [B, N, H]
            f_mask: Feature mask used [B, T, N]
            s_mask: Structure mask used [N, N]
        """
        B, T, N, D = x.shape

        # Get raw features (first channel only for masking)
        raw_x = x[..., :1].clone() if D > 1 else x.clone()

        # Feature masking
        x_masked, f_mask = self.feature_masking(
            raw_x, mask_f, mask=f_mask, patch_length=self.patch_length
        )

        # Feature embedding with mask token replacement
        embed_x = self.to_feat_embedding(x_masked)  # [B, T, N, H]
        mask_token_expanded = repeat(self.mask_token, 'd -> b t n d', b=B, t=T, n=N)
        embed_x = embed_x * f_mask.unsqueeze(-1) + mask_token_expanded * (1 - f_mask.unsqueeze(-1))

        # Structure masking
        support = self.get_support()
        s_mask = self.structure_masking(support, mask_s, mask=s_mask, **kwargs)

        # Encode
        embedding = self._encode_with_backbone(embed_x, s_mask, **kwargs)

        # Summary: last timestep
        summary = embedding[:, -1, :, :]  # [B, N, H]

        return embedding, summary, f_mask, s_mask

    def _encode_with_backbone(
        self,
        x: torch.Tensor,
        s_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode input through backbone.

        Args:
            x: Embedded input [B, T, N, H]
            s_mask: Structure mask [N, N]

        Returns:
            embedding: Encoder output [B, T, N, H]
        """
        # Check if encoder supports structure mask
        if hasattr(self.encoder, 'forward_with_mask'):
            embedding = self.encoder.forward_with_mask(x, s_mask, **kwargs)
        else:
            # Default encoding without mask
            embedding = self.encoder(x, **kwargs)

        return embedding

    def decode_structure(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Decode structure (adjacency) from summary.

        Args:
            summary: Summary embedding [B, N, H]

        Returns:
            adj: Reconstructed adjacency [B, N, N]
        """
        return self.structure_decoder(summary)

    def decode_feature(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Decode features from summary.

        Args:
            summary: Summary embedding [B, N, H]

        Returns:
            reconstruction: Reconstructed features [B, T, N, D]
        """
        return self.feature_decoder(summary)

    def forward_s_loss(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        s_mask: torch.Tensor,
        loss_type: str = 'cls_boost',
    ) -> torch.Tensor:
        """
        Compute structure reconstruction loss.

        Args:
            target: Target support [N, N]
            pred: Predicted adjacency [B, N, N]
            s_mask: Structure mask [N, N] where 0 = masked
            loss_type: Loss type ('cls_boost' uses BCE on masked positions)

        Returns:
            loss: Structure loss scalar
        """
        B = pred.shape[0]

        if loss_type == 'cls_boost':
            # Target: all ones (predict connectivity)
            target = torch.ones_like(target)
            target = repeat(target, 'm n -> b m n', b=B)

            # Inverse mask: 1 where masked (positions to reconstruct)
            inv_mask = repeat(1 - s_mask, 'm n -> b m n', b=B)

            if torch.sum(inv_mask) != 0:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * inv_mask
                loss = torch.sum(diff) / torch.sum(inv_mask)
            else:
                diff = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
                loss = torch.mean(diff)
        else:
            raise ValueError(f"Unknown structure loss type: {loss_type}")

        return loss

    def forward_f_loss(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        f_mask: torch.Tensor,
        loss_type: str = 'reg_l1',
    ) -> torch.Tensor:
        """
        Compute feature reconstruction loss.

        Args:
            target: Target features [B, T, N, D]
            pred: Predicted features [B, T, N, D]
            f_mask: Feature mask [B, T, N] where 0 = masked
            loss_type: Loss type ('reg_l1' uses MAE on masked positions)

        Returns:
            loss: Feature loss scalar
        """
        # Use only first channel for reconstruction
        target = target[..., :1]
        pred = pred[..., :1] if pred.shape[-1] > 1 else pred

        if loss_type == 'reg_l1':
            target = target.squeeze(-1)  # [B, T, N]
            pred = pred.squeeze(-1)  # [B, T, N]

            # Inverse mask: 1 where masked (positions to reconstruct)
            inv_mask = 1 - f_mask

            if torch.sum(inv_mask) != 0:
                diff = torch.abs(pred - target) * inv_mask
                loss = torch.sum(diff) / torch.sum(inv_mask)
            else:
                diff = torch.abs(pred - target)
                loss = torch.mean(diff)
        else:
            raise ValueError(f"Unknown feature loss type: {loss_type}")

        return loss

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """
        Forward pass for pre-training.

        Args:
            history_data: Input tensor [B, T, N, D]
            future_data: Not used in pre-training
            batch_seen: Batch counter
            epoch: Current epoch
            train: Training mode flag

        Returns:
            Dictionary containing:
                - loss: Total loss
                - s_loss: Structure loss
                - f_loss: Feature loss
                - f_mask: Feature mask
                - s_mask: Structure mask
                - embedding: Encoder embedding
                - summary: Summary embedding
        """
        x = history_data

        # Get masking ratios (can be overridden in kwargs)
        mask_s = kwargs.get('mask_s', self.mask_s_ratio if train else 0)
        mask_f = kwargs.get('mask_f', self.mask_f_ratio if train else 0)

        # Encode with masking
        embedding, summary, f_mask, s_mask = self.encode(
            x, mask_s=mask_s, mask_f=mask_f, **kwargs
        )

        # Decode
        recon_s = self.decode_structure(summary)  # [B, N, N]
        recon_f = self.decode_feature(summary)  # [B, T, N, D]

        # Compute losses
        sl_type = kwargs.get('sl_type', 'cls_boost')
        fl_type = kwargs.get('fl_type', 'reg_l1')

        s_loss = self.forward_s_loss(self.get_support(), recon_s, s_mask, sl_type)
        f_loss = self.forward_f_loss(x, recon_f, f_mask, fl_type)

        # Weighted sum
        sl_weight = kwargs.get('sl_weight', self.sl_weight)
        fl_weight = kwargs.get('fl_weight', self.fl_weight)
        loss = sl_weight * s_loss + fl_weight * f_loss

        return {
            'loss': loss,
            's_loss': s_loss,
            'f_loss': f_loss,
            'f_mask': f_mask,
            's_mask': s_mask,
            'embedding': embedding,
            'summary': summary,
            'reconstruction': recon_f,
        }


class STMAEForecaster(nn.Module):
    """
    STMAE Forecaster for fine-tuning.

    Uses pre-trained encoder with a forecasting decoder.
    """

    def __init__(
        self,
        encoder: STMAE,
        output_len: int,
        output_dim: int = 1,
        use_mlp_decoder: bool = False,
        freeze_encoder: bool = False,
    ):
        """
        Args:
            encoder: Pre-trained STMAE encoder
            output_len: Prediction horizon
            output_dim: Output feature dimension
            use_mlp_decoder: Use MLP decoder (more capacity)
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()

        self.encoder = encoder
        self.output_len = output_len
        self.output_dim = output_dim
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Forecasting decoder
        self.decoder = ForecastingDecoder(
            hidden_dim=encoder.hidden_dim,
            output_dim=output_dim,
            horizon=output_len,
            use_mlp=use_mlp_decoder,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for forecasting.

        Args:
            history_data: Input tensor [B, T, N, D]
            future_data: Not used
            batch_seen: Batch counter
            epoch: Current epoch
            train: Training mode flag

        Returns:
            prediction: Future predictions [B, horizon, N, D]
        """
        # Encode without masking
        _, summary, _, _ = self.encoder.encode(history_data, mask_s=0, mask_f=0, **kwargs)

        # Decode for forecasting
        prediction = self.decoder(summary)

        return prediction


class DefaultEncoder(nn.Module):
    """
    Default GRU-based encoder when no backbone is specified.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim * num_nodes,
            hidden_size=hidden_dim * num_nodes,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, N, H]

        Returns:
            output: Encoded output [B, T, N, H]
        """
        B, T, N, H = x.shape
        # Reshape for GRU: [B, T, N*H]
        x_flat = x.reshape(B, T, -1)
        output, _ = self.gru(x_flat)
        # Reshape back: [B, T, N, H]
        output = output.reshape(B, T, N, H)
        return output

    def forward_with_mask(self, x: torch.Tensor, s_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward with structure mask (ignored in default encoder)."""
        return self.forward(x, **kwargs)
