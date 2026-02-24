"""
Test-Time Adaptation Runner.

Supports two TTA strategies:
1. Ensemble: Multiple forward passes with dropout, average predictions
2. Adaptation: Self-supervised adaptation (node masking reconstruction)

Config:
    CFG.TTA = {
        'ENABLED': True,           # Enable TTA at test time
        'MODE': 'ensemble',        # 'ensemble', 'adaptation', or 'both'

        # Ensemble settings
        'ENSEMBLE_RUNS': 5,        # Number of forward passes for ensemble
        'ENSEMBLE_DROPOUT': 0.1,   # Dropout rate for ensemble diversity

        # Adaptation settings
        'ADAPT_STEPS': 1,          # Gradient steps per sample (1-3 recommended)
        'ADAPT_LR': 1e-4,          # Adaptation learning rate (keep small!)
        'MASK_RATIO': 0.15,        # Node masking ratio for self-supervised signal
        'ADAPT_LAYERS': 'all',     # 'all', 'last', or 'norm' (which layers to adapt)
    }
"""

import copy
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representation_learning_runner import RepresentationLearningRunner


# Default TTA configuration
DEFAULT_TTA_CONFIG = {
    'ENABLED': False,
    'MODE': 'ensemble',  # 'ensemble', 'adaptation', 'both'

    # Ensemble settings
    'ENSEMBLE_RUNS': 5,
    'ENSEMBLE_DROPOUT': 0.1,

    # Adaptation settings
    'ADAPT_STEPS': 1,
    'ADAPT_LR': 1e-4,
    'MASK_RATIO': 0.15,
    'ADAPT_LAYERS': 'all',  # 'all', 'last', 'norm'
}


class TTARunner(RepresentationLearningRunner):
    """
    Test-Time Adaptation Runner.

    Extends RepresentationLearningRunner with TTA capabilities:
    - Ensemble: Run multiple forward passes with stochastic elements
    - Adaptation: Adapt encoder on each test sample using self-supervision
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Parse TTA config with defaults
        tta_cfg = cfg.get('TTA', {})
        self.tta_config = {**DEFAULT_TTA_CONFIG, **tta_cfg}

        self.tta_enabled = self.tta_config['ENABLED']
        self.tta_mode = self.tta_config['MODE']

        # Ensemble settings
        self.ensemble_runs = self.tta_config['ENSEMBLE_RUNS']
        self.ensemble_dropout = self.tta_config['ENSEMBLE_DROPOUT']

        # Adaptation settings
        self.adapt_steps = self.tta_config['ADAPT_STEPS']
        self.adapt_lr = self.tta_config['ADAPT_LR']
        self.mask_ratio = self.tta_config['MASK_RATIO']
        self.adapt_layers = self.tta_config['ADAPT_LAYERS']

        # Create dropout layer for ensemble
        self.tta_dropout = nn.Dropout(p=self.ensemble_dropout)

        if self.tta_enabled:
            self.logger.info(f"TTA enabled: mode={self.tta_mode}")
            if self.tta_mode in ['ensemble', 'both']:
                self.logger.info(f"  Ensemble: runs={self.ensemble_runs}, dropout={self.ensemble_dropout}")
            if self.tta_mode in ['adaptation', 'both']:
                self.logger.info(f"  Adaptation: steps={self.adapt_steps}, lr={self.adapt_lr}, mask={self.mask_ratio}")

    def forward(
        self,
        data: Dict,
        epoch: Optional[int] = None,
        iter_num: Optional[int] = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """Forward pass with optional TTA at test time."""
        # Training or TTA disabled: use standard forward
        if train or not self.tta_enabled:
            return super().forward(data, epoch, iter_num, train, **kwargs)

        # Test time with TTA
        if self.tta_mode == 'ensemble':
            return self._forward_ensemble(data)
        elif self.tta_mode == 'adaptation':
            return self._forward_adaptation(data)
        elif self.tta_mode == 'both':
            return self._forward_both(data)
        else:
            return super().forward(data, epoch, iter_num, train, **kwargs)

    # ==================== Ensemble TTA ====================

    def _forward_ensemble(self, data: Dict) -> Dict:
        """
        Ensemble TTA: Multiple forward passes with dropout, average results.

        Simple but effective - uses stochastic dropout to create diverse predictions.
        """
        data = self.preprocessing(data)
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        predictions = []

        # Run multiple forward passes with dropout
        for _ in range(self.ensemble_runs):
            # Encode with dropout diversity
            with torch.no_grad():
                encoded = self.encoder.encode(history_data, self.adj)
                # Apply dropout for diversity (train mode enables dropout)
                self.tta_dropout.train()
                encoded = self.tta_dropout(encoded)

            # Include tod/dow if needed
            if self.include_tod_dow:
                placeholder = encoded[..., 0:1]
                tod = history_data[..., self.tod_idx:self.tod_idx+1]
                dow = history_data[..., self.dow_idx:self.dow_idx+1]
                encoded_rest = encoded[..., 1:]
                encoded = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)

            # Get prediction
            with torch.no_grad():
                pred = self.model(
                    history_data=encoded,
                    future_data=future_data_4_dec,
                    batch_seen=None,
                    epoch=None,
                    train=False
                )
                if isinstance(pred, dict):
                    pred = pred['prediction']
                predictions.append(pred)

        # Ensemble: average predictions
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)

        result = {
            'prediction': ensemble_pred,
            'target': self.select_target_features(future_data),
            'inputs': self.select_target_features(history_data),
        }
        return self.postprocessing(result)

    # ==================== Adaptation TTA ====================

    def _forward_adaptation(self, data: Dict) -> Dict:
        """
        Adaptation TTA: Self-supervised adaptation per test sample.

        Uses node masking reconstruction as self-supervised signal.
        Adapts encoder, then predicts, then restores encoder state.
        """
        data = self.preprocessing(data)
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # 1. Save encoder state
        encoder_state = copy.deepcopy(self.encoder.state_dict())

        # 2. Get parameters to adapt
        adapt_params = self._get_adapt_params()

        # 3. Self-supervised adaptation
        if adapt_params:
            self.encoder.train()
            tta_optimizer = torch.optim.SGD(adapt_params, lr=self.adapt_lr, momentum=0.9)

            for _ in range(self.adapt_steps):
                # Create masked input
                masked_data, mask = self._random_mask_nodes(history_data)

                # Forward both masked and original
                encoded_masked = self.encoder.encode(masked_data, self.adj)
                with torch.no_grad():
                    encoded_original = self.encoder.encode(history_data, self.adj)

                # Reconstruction loss on masked positions
                loss = F.mse_loss(
                    encoded_masked[mask.expand_as(encoded_masked)],
                    encoded_original[mask.expand_as(encoded_original)]
                )

                tta_optimizer.zero_grad()
                loss.backward()
                tta_optimizer.step()

        # 4. Predict with adapted encoder
        self.encoder.eval()
        with torch.no_grad():
            encoded = self.encoder.encode(history_data, self.adj)

            if self.include_tod_dow:
                placeholder = encoded[..., 0:1]
                tod = history_data[..., self.tod_idx:self.tod_idx+1]
                dow = history_data[..., self.dow_idx:self.dow_idx+1]
                encoded_rest = encoded[..., 1:]
                encoded = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)

            pred = self.model(
                history_data=encoded,
                future_data=future_data_4_dec,
                batch_seen=None,
                epoch=None,
                train=False
            )
            if isinstance(pred, dict):
                pred = pred['prediction']

        # 5. Restore encoder state
        self.encoder.load_state_dict(encoder_state)

        result = {
            'prediction': pred,
            'target': self.select_target_features(future_data),
            'inputs': self.select_target_features(history_data),
        }
        return self.postprocessing(result)

    def _forward_both(self, data: Dict) -> Dict:
        """
        Combined TTA: Adaptation + Ensemble.

        First adapts encoder, then runs ensemble predictions.
        """
        data = self.preprocessing(data)
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # 1. Save encoder state
        encoder_state = copy.deepcopy(self.encoder.state_dict())

        # 2. Adaptation phase
        adapt_params = self._get_adapt_params()
        if adapt_params:
            self.encoder.train()
            tta_optimizer = torch.optim.SGD(adapt_params, lr=self.adapt_lr, momentum=0.9)

            for _ in range(self.adapt_steps):
                masked_data, mask = self._random_mask_nodes(history_data)
                encoded_masked = self.encoder.encode(masked_data, self.adj)
                with torch.no_grad():
                    encoded_original = self.encoder.encode(history_data, self.adj)

                loss = F.mse_loss(
                    encoded_masked[mask.expand_as(encoded_masked)],
                    encoded_original[mask.expand_as(encoded_original)]
                )

                tta_optimizer.zero_grad()
                loss.backward()
                tta_optimizer.step()

        # 3. Ensemble phase with adapted encoder
        self.encoder.eval()
        predictions = []

        for _ in range(self.ensemble_runs):
            with torch.no_grad():
                encoded = self.encoder.encode(history_data, self.adj)
                self.tta_dropout.train()
                encoded = self.tta_dropout(encoded)

                if self.include_tod_dow:
                    placeholder = encoded[..., 0:1]
                    tod = history_data[..., self.tod_idx:self.tod_idx+1]
                    dow = history_data[..., self.dow_idx:self.dow_idx+1]
                    encoded_rest = encoded[..., 1:]
                    encoded = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)

                pred = self.model(
                    history_data=encoded,
                    future_data=future_data_4_dec,
                    batch_seen=None,
                    epoch=None,
                    train=False
                )
                if isinstance(pred, dict):
                    pred = pred['prediction']
                predictions.append(pred)

        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)

        # 4. Restore encoder state
        self.encoder.load_state_dict(encoder_state)

        result = {
            'prediction': ensemble_pred,
            'target': self.select_target_features(future_data),
            'inputs': self.select_target_features(history_data),
        }
        return self.postprocessing(result)

    # ==================== Helper Methods ====================

    def _get_adapt_params(self) -> List[torch.nn.Parameter]:
        """Get parameters to adapt based on config."""
        if self.adapt_layers == 'all':
            return list(self.encoder.parameters())
        elif self.adapt_layers == 'last':
            # Only last layer (if encoder has layers attribute)
            if hasattr(self.encoder, 'layers') and len(self.encoder.layers) > 0:
                return list(self.encoder.layers[-1].parameters())
            return list(self.encoder.parameters())
        elif self.adapt_layers == 'norm':
            # Only normalization layers
            params = []
            for name, param in self.encoder.named_parameters():
                if 'norm' in name.lower() or 'ln' in name.lower():
                    params.append(param)
            return params if params else list(self.encoder.parameters())
        else:
            return list(self.encoder.parameters())

    def _random_mask_nodes(self, x: torch.Tensor, ratio: float = None) -> tuple:
        """
        Random node masking for self-supervised signal.

        Args:
            x: Input tensor [B, T, N, C]
            ratio: Masking ratio (default: self.mask_ratio)

        Returns:
            masked_x: Tensor with some nodes zeroed out
            mask: Boolean mask indicating masked positions [B, 1, N, 1]
        """
        if ratio is None:
            ratio = self.mask_ratio

        B, T, N, C = x.shape

        # Create node-level mask (same mask across time)
        mask = torch.rand(B, 1, N, 1, device=x.device) < ratio

        # Apply mask
        masked_x = x.clone()
        mask_expanded = mask.expand(B, T, N, C)
        masked_x[mask_expanded] = 0

        return masked_x, mask

    def test(self, train_epoch: Optional[int] = None,
             save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test with TTA info logging."""
        if self.tta_enabled:
            self.logger.info(f"Testing with TTA mode: {self.tta_mode}")

        return super().test(train_epoch, save_metrics, save_results)
