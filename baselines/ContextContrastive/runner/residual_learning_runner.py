"""
Residual Learning Runner.

Trains a lightweight ResidualMLP to predict Δŷ that corrects a frozen backbone's prediction.

Supports two modes:
  1. With encoder (CFG.ENCODER provided):
     Input → Encoder (frozen) → z_context, z_self → concat → ResidualMLP → Δŷ
  2. Without encoder (no CFG.ENCODER / raw input ablation):
     Input (raw features) → ResidualMLP → Δŷ

Final prediction: ŷ = ŷ_backbone + Δŷ

Config:
    CFG.BACKBONE = {
        'arch': STAEformer,
        'param': {...},
        'ckpt_path': '...',
        'feature_indices': [0, 1, 2],   # indices within FORWARD_FEATURES for backbone
    }

    # Optional: omit for raw input ablation
    CFG.ENCODER = {
        'type': 'DisentangledEncoder',
        'source': 'pretrained',
        'ckpt_path': '...',
        'd_model': 64,
        ...
    }
"""

import glob
from typing import Dict, Optional

import torch
import torch.nn as nn

from basicts.runners import SimpleTimeSeriesForecastingRunner
from ..arch.base_encoder import build_encoder


class ResidualLearningRunner(SimpleTimeSeriesForecastingRunner):
    """Runner that trains a residual correction on top of a frozen backbone."""

    def __init__(self, cfg: Dict):
        # Extract configs before super().__init__
        backbone_cfg = cfg['BACKBONE']
        self.backbone_cfg = backbone_cfg
        self.backbone_feature_indices = backbone_cfg.get('feature_indices', [0, 1, 2])

        # Encoder is optional (None = raw input ablation)
        encoder_cfg = cfg.get('ENCODER')
        self.encoder_cfg = encoder_cfg
        self.use_encoder = encoder_cfg is not None

        if self.use_encoder:
            self.encoder = build_encoder(encoder_cfg)
            self._load_encoder_weights(encoder_cfg['ckpt_path'])
        else:
            self.encoder = None

        # super().__init__ builds self.model (ResidualMLP)
        super().__init__(cfg)

        device = next(self.model.parameters()).device

        # Freeze encoder if present
        if self.use_encoder:
            self.encoder = self.encoder.to(device)
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Build and freeze backbone
        backbone_arch = backbone_cfg['arch']
        backbone_param = backbone_cfg['param']
        self.backbone = backbone_arch(**backbone_param).to(device)
        self._load_backbone_weights(backbone_cfg['ckpt_path'])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Logging
        self.logger.info(f"Backbone: {backbone_arch.__name__} (frozen)")
        if self.use_encoder:
            self.logger.info(f"Encoder: {encoder_cfg.get('type')} (frozen)")
        else:
            self.logger.info("Encoder: None (raw input mode)")
        self.logger.info(f"Trainable: ResidualMLP ({sum(p.numel() for p in self.model.parameters())} params)")

    def _resolve_ckpt_path(self, ckpt_path: str) -> str:
        """Resolve glob patterns in checkpoint path."""
        if '*' in ckpt_path:
            matches = glob.glob(ckpt_path)
            if not matches:
                raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")
            ckpt_path = sorted(matches)[-1]
        return ckpt_path

    def _load_encoder_weights(self, ckpt_path: str):
        """Load pretrained encoder weights."""
        ckpt_path = self._resolve_ckpt_path(ckpt_path)
        print(f"Loading encoder from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        missing, unexpected = self.encoder.load_pretrained_weights(state_dict, strict=False)
        loaded = len(self.encoder.state_dict()) - len(missing)
        print(f"  Loaded {loaded}/{len(self.encoder.state_dict())} encoder parameters")

    def _load_backbone_weights(self, ckpt_path: str):
        """Load backbone model weights."""
        ckpt_path = self._resolve_ckpt_path(ckpt_path)
        print(f"Loading backbone from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.backbone.load_state_dict(state_dict)
        print(f"  Loaded {len(state_dict)} backbone parameters")

    def _get_residual_input(self, history_data: torch.Tensor) -> torch.Tensor:
        """Get input for ResidualMLP: encoder output or raw features."""
        if self.use_encoder:
            with torch.no_grad():
                z_context, z_self = self.encoder.encode_disentangled(history_data)
            return torch.cat([z_context, z_self], dim=-1)  # [B, T, N, d_model*2]
        else:
            return history_data  # [B, T, N, C] raw features

    def forward(
        self,
        data: Dict,
        epoch: Optional[int] = None,
        iter_num: Optional[int] = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """Forward pass: backbone prediction + residual correction."""
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features (FORWARD_FEATURES applied here)
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # 1. Backbone forward (frozen)
        backbone_input = history_data[..., self.backbone_feature_indices]
        backbone_future = future_data_4_dec[..., self.backbone_feature_indices]
        with torch.no_grad():
            backbone_output = self.backbone(
                history_data=backbone_input,
                future_data=backbone_future,
                batch_seen=None, epoch=None, train=False
            )
            y_backbone = backbone_output['prediction']

        # 2. Get residual input (encoder output or raw features)
        residual_input = self._get_residual_input(history_data)

        # 3. ResidualMLP forward (trainable)
        residual_output = self.model(
            history_data=residual_input,
            future_data=None,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )
        delta_y = residual_output['prediction']  # [B, T_out, N, 1]

        # 4. Final prediction = backbone + residual
        prediction = y_backbone + delta_y

        model_return = {
            'prediction': prediction,
            'inputs': self.select_target_features(history_data),
            'target': self.select_target_features(future_data),
        }

        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], \
            f"Shape mismatch: expected [B={batch_size}, L={length}, N={num_nodes}], got {prediction.shape}"

        model_return = self.postprocessing(model_return)
        return model_return
