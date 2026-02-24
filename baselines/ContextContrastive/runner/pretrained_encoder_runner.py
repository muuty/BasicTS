"""
Stage 2 Runner: Uses pre-trained encoder to transform input data before downstream model.

Usage:
    1. Pre-train encoder using pretrain.py (Stage 1)
    2. Use this runner with downstream model config (Stage 2)
"""
import glob
from typing import Dict, Optional

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner
from ..arch import ContextAwareEncoder


class PretrainedEncoderRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner that applies a pre-trained encoder to input data before passing to downstream model.

    The downstream model (STGCN, STAEformer, etc.) receives encoded representations
    instead of raw features.

    Config requirements:
        CFG.PRETRAINED_ENCODER: dict with keys:
            - ckpt_path: path to pre-trained checkpoint
            - d_model: encoder output dimension
            - input_dim: original input dimension (before encoding)
            - num_layers: number of transformer layers
            - nhead: number of attention heads
            - dropout: dropout rate
            - include_tod_dow: (optional) if True, output [placeholder, tod, dow, encoded]
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Load pre-trained encoder config
        encoder_cfg = cfg.get('PRETRAINED_ENCODER', {})
        if not encoder_cfg:
            raise ValueError("PRETRAINED_ENCODER config is required for PretrainedEncoderRunner")

        # Build and load encoder
        self.encoder = self._build_encoder(encoder_cfg)
        self._load_encoder_weights(encoder_cfg.get('ckpt_path'))

        # Option to include tod/dow for downstream models
        self.include_tod_dow = encoder_cfg.get('include_tod_dow', False)

        # Freeze encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.logger.info(f"Loaded pre-trained encoder from {encoder_cfg.get('ckpt_path')}")
        self.logger.info(f"Encoder output dimension: {encoder_cfg.get('d_model')}")
        self.logger.info(f"Include tod/dow in output: {self.include_tod_dow}")

    def _build_encoder(self, encoder_cfg: Dict) -> ContextAwareEncoder:
        """Build encoder from config."""
        return ContextAwareEncoder(
            input_dim=encoder_cfg.get('input_dim', 3),
            d_model=encoder_cfg.get('d_model', 64),
            num_layers=encoder_cfg.get('num_layers', 2),
            nhead=encoder_cfg.get('nhead', 4),
            dropout=encoder_cfg.get('dropout', 0.1),
        )

    def _load_encoder_weights(self, ckpt_path: str):
        """Load encoder weights from checkpoint."""
        if not ckpt_path:
            raise ValueError("ckpt_path is required to load pre-trained encoder")

        # Resolve glob pattern if present
        if '*' in ckpt_path:
            matches = glob.glob(ckpt_path)
            if not matches:
                raise FileNotFoundError(f"No checkpoint found matching pattern: {ckpt_path}")
            ckpt_path = sorted(matches)[-1]
            self.logger.info(f"Resolved checkpoint path: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Filter for encoder weights only
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value

        if not encoder_state_dict:
            raise ValueError(f"No encoder weights found in checkpoint: {ckpt_path}")

        # Handle legacy key names (old: temporal_input_proj, new: temporal_encoder.input_proj)
        remapped_state_dict = {}
        for key, value in encoder_state_dict.items():
            if key.startswith('temporal_input_proj'):
                new_key = key.replace('temporal_input_proj', 'temporal_encoder.input_proj')
            elif key.startswith('temporal_transformer'):
                new_key = key.replace('temporal_transformer', 'temporal_encoder.transformer')
            else:
                new_key = key
            remapped_state_dict[new_key] = value

        self.encoder.load_state_dict(remapped_state_dict)
        self.logger.info(f"Loaded {len(encoder_state_dict)} encoder parameters")

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Forward pass with pre-trained encoder transformation.
        """
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Encode history_data with pre-trained encoder
        with torch.no_grad():
            self.encoder.to(history_data.device)
            encoded_history = self.encoder(history_data)  # [B, L, N, D]

            # Optionally include tod/dow for downstream models
            if self.include_tod_dow:
                placeholder = encoded_history[..., 0:1]
                tod = history_data[..., 1:2]
                dow = history_data[..., 2:3]
                encoded_rest = encoded_history[..., 1:]
                encoded_history = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)

        # Forward pass through downstream model
        model_return = self.model(
            history_data=encoded_history,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)
        return model_return
