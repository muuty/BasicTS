"""
Encoder-Backbone Wrapper Model.

Combines a pre-trained encoder with a prediction backbone (STGCN, STAEformer, etc.)
The encoder is frozen and provides representations to the backbone.

Reference: design.md - Section 2.3 (Stage 2: Prediction Fine-tuning)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class EncoderBackboneModel(nn.Module):
    """
    Wrapper that combines a pre-trained encoder with a prediction backbone.

    The encoder transforms raw input into learned representations,
    which are then fed to the backbone for prediction.

    Args:
        encoder: Pre-trained encoder module (e.g., ContextAwareSTEncoder)
        backbone: Prediction backbone module (e.g., STGCN, STAEformer)
        freeze_encoder: Whether to freeze encoder parameters (default: True)
        projection_dim: If set, add a projection layer between encoder and backbone
                       to match dimensions (default: None, no projection)
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        freeze_encoder: bool = True,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.backbone = backbone
        self.freeze_encoder = freeze_encoder

        # Get encoder output dimension
        if hasattr(encoder, 'get_output_dim'):
            encoder_dim = encoder.get_output_dim()
        elif hasattr(encoder, 'd_model'):
            encoder_dim = encoder.d_model
        else:
            raise ValueError("Encoder must have 'get_output_dim()' method or 'd_model' attribute")

        # Optional projection layer
        if projection_dim is not None and projection_dim != encoder_dim:
            self.projection = nn.Sequential(
                nn.Linear(encoder_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
            )
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = encoder_dim

        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and backbone.

        Args:
            history_data: Input tensor of shape (B, T, N, C)
            future_data: Future data tensor (passed to backbone)
            batch_seen: Number of batches seen (passed to backbone)
            epoch: Current epoch (passed to backbone)
            train: Training mode flag
            **kwargs: Additional arguments (may include time_of_day_idx, day_of_week_idx, etc.)

        Returns:
            Dict containing 'prediction' and optionally other outputs
        """
        # Extract context indices from kwargs if available
        time_of_day_idx = kwargs.get('time_of_day_idx', None)
        day_of_week_idx = kwargs.get('day_of_week_idx', None)
        node_idx = kwargs.get('node_idx', None)

        # Encode input
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder(
                    history_data,
                    time_of_day_idx=time_of_day_idx,
                    day_of_week_idx=day_of_week_idx,
                    node_idx=node_idx,
                )
        else:
            z = self.encoder(
                history_data,
                time_of_day_idx=time_of_day_idx,
                day_of_week_idx=day_of_week_idx,
                node_idx=node_idx,
            )

        # Optional projection
        if self.projection is not None:
            z = self.projection(z)

        # Pass encoded representation to backbone
        # Note: backbone expects input in shape (B, T, N, C)
        output = self.backbone(
            history_data=z,
            future_data=future_data,
            batch_seen=batch_seen,
            epoch=epoch,
            train=train,
            **kwargs
        )

        # Handle different backbone output formats
        if isinstance(output, torch.Tensor):
            return {'prediction': output}
        elif isinstance(output, dict):
            return output
        else:
            raise ValueError(f"Unexpected backbone output type: {type(output)}")

    def train(self, mode: bool = True):
        """Set training mode, keeping encoder frozen if specified."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def get_encoder_params(self):
        """Get encoder parameters (for optimizer if not frozen)."""
        return self.encoder.parameters()

    def get_backbone_params(self):
        """Get backbone parameters (for optimizer)."""
        return self.backbone.parameters()

    def get_trainable_params(self):
        """Get all trainable parameters."""
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        projection_params = 0
        projection_trainable = 0
        if self.projection is not None:
            projection_params = sum(p.numel() for p in self.projection.parameters())
            projection_trainable = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)

        return {
            'encoder_total': encoder_params,
            'encoder_trainable': encoder_trainable,
            'backbone_total': backbone_params,
            'backbone_trainable': backbone_trainable,
            'projection_total': projection_params,
            'projection_trainable': projection_trainable,
            'total': encoder_params + backbone_params + projection_params,
            'total_trainable': encoder_trainable + backbone_trainable + projection_trainable,
        }


def build_encoder_backbone_model(
    encoder_cfg: Dict[str, Any],
    backbone_cfg: Dict[str, Any],
    encoder_ckpt: Optional[str] = None,
    freeze_encoder: bool = True,
    projection_dim: Optional[int] = None,
) -> EncoderBackboneModel:
    """
    Factory function to build EncoderBackboneModel from configs.

    Args:
        encoder_cfg: Encoder configuration dict with 'name' and 'param' keys
        backbone_cfg: Backbone configuration dict with 'name' and 'param' keys
        encoder_ckpt: Path to pre-trained encoder checkpoint
        freeze_encoder: Whether to freeze encoder
        projection_dim: Optional projection dimension

    Returns:
        EncoderBackboneModel instance
    """
    from basicts.utils import load_model

    # Build encoder
    encoder_name = encoder_cfg['name']
    encoder_params = encoder_cfg.get('param', {})

    if encoder_name == 'ContextAwareSTEncoder':
        from encoder import ContextAwareSTEncoder
        encoder = ContextAwareSTEncoder(**encoder_params)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    # Load pre-trained encoder weights if provided
    if encoder_ckpt is not None:
        ckpt = torch.load(encoder_ckpt, map_location='cpu')
        if 'model_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            encoder.load_state_dict(ckpt['state_dict'])
        else:
            encoder.load_state_dict(ckpt)
        print(f"Loaded encoder checkpoint from: {encoder_ckpt}")

    # Build backbone
    backbone_name = backbone_cfg['name']
    backbone_params = backbone_cfg.get('param', {})

    # Import backbone dynamically
    if backbone_name == 'STGCNChebGraphConv':
        from baselines.STGCN.arch import STGCNChebGraphConv
        backbone = STGCNChebGraphConv(**backbone_params)
    elif backbone_name == 'STAEformer':
        from baselines.STAEformer.arch import STAEformer
        backbone = STAEformer(**backbone_params)
    elif backbone_name == 'AGCRN':
        from baselines.AGCRN.arch import AGCRN
        backbone = AGCRN(**backbone_params)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Create wrapper
    model = EncoderBackboneModel(
        encoder=encoder,
        backbone=backbone,
        freeze_encoder=freeze_encoder,
        projection_dim=projection_dim,
    )

    return model
