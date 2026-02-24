"""
Representation Learning Runner.

Unified runner for all representation learning experiments:
- Any encoder type (Transformer, DilatedConv, SpatioTemporal, MaskedAutoEncoder)
- Any mode (pretrained, scratch, frozen)
- Any downstream model (STAEformer, STGCN, etc.)
- Test-Time Adaptation (TTA) support

Config:
    CFG.ENCODER = {
        'type': 'TransformerEncoder',  # or 'DilatedConvEncoder', 'SpatioTemporalEncoder', 'MaskedAutoEncoder'
        'source': 'pretrained',        # 'pretrained' or 'scratch'
        'freeze': False,               # True: frozen encoder, False: fine-tune
        'ckpt_path': '...',            # required if source='pretrained'
        'lr': 1e-5,                    # encoder-specific learning rate (optional)
        'tod_idx': 1,                  # index of time-of-day feature (default: 1)
        'dow_idx': 2,                  # index of day-of-week feature (default: 2)
        # Encoder architecture params:
        'd_model': 64,
        'num_layers': 2,
        'nhead': 4,
        'dropout': 0.1,
        ...
    }

    # Optional TTA config (disabled by default)
    CFG.TTA = {
        'ENABLED': True,              # Enable TTA at test time
        'MODE': 'ensemble',           # 'ensemble', 'adaptation', or 'both'
        'ENSEMBLE_RUNS': 5,           # Forward passes for ensemble
        'ENSEMBLE_DROPOUT': 0.1,      # Dropout rate for diversity
        'ADAPT_STEPS': 1,             # Gradient steps for adaptation
        'ADAPT_LR': 1e-4,             # Adaptation learning rate
        'MASK_RATIO': 0.15,           # Node masking ratio
        'ADAPT_LAYERS': 'all',        # 'all', 'last', or 'norm'
    }
"""

import os
import copy
import glob
import json
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from easytorch.core.checkpoint import save_ckpt, load_ckpt, backup_last_ckpt

from basicts.runners import SimpleTimeSeriesForecastingRunner
from ..arch.base_encoder import build_encoder

# Constants
DEFAULT_TOD_IDX = 1
DEFAULT_DOW_IDX = 2
# Incident occurs at last timestep of input window (input_len=12, so offset=11)
INCIDENT_SLOT_OFFSET = 11


class RepresentationLearningRunner(SimpleTimeSeriesForecastingRunner):
    """
    Unified runner for representation learning experiments.

    Supports:
    - Multiple encoder types via unified interface
    - Pretrained or from-scratch encoder training
    - Frozen or fine-tuned encoder
    - Discriminative learning rates
    - Proper checkpoint save/load for encoder weights
    """

    def __init__(self, cfg: Dict):
        # Extract encoder config before super().__init__
        encoder_cfg = cfg.get('ENCODER', {})
        if not encoder_cfg:
            raise ValueError("ENCODER config is required")

        self.encoder_cfg = encoder_cfg
        self.encoder_source = encoder_cfg.get('source', 'scratch')  # 'pretrained' or 'scratch'
        self.encoder_freeze = encoder_cfg.get('freeze', False)
        self.encoder_lr = encoder_cfg.get('lr', None)
        self.include_tod_dow = encoder_cfg.get('include_tod_dow', False)
        self.tod_idx = encoder_cfg.get('tod_idx', DEFAULT_TOD_IDX)
        self.dow_idx = encoder_cfg.get('dow_idx', DEFAULT_DOW_IDX)
        self.pass_through_indices = encoder_cfg.get('pass_through_indices', None)
        self.downstream_features = encoder_cfg.get('downstream_features', None)
        self.output_reliability = encoder_cfg.get('output_reliability', False)
        self.mask_channel_idx = cfg.get('MODEL', {}).get('MASK_CHANNEL_IDX', None)

        # Build encoder (encoder loads its own adj if adj_path in config)
        self.encoder = build_encoder(encoder_cfg)
        self._encoder_on_device = False

        # Load pretrained weights if specified
        if self.encoder_source == 'pretrained':
            ckpt_path = encoder_cfg.get('ckpt_path')
            if not ckpt_path:
                raise ValueError("ckpt_path required for pretrained encoder")
            self._load_pretrained_encoder(ckpt_path)

        # Call parent init
        super().__init__(cfg)

        # Move encoder to device
        device = next(self.model.parameters()).device
        self.encoder = self.encoder.to(device)
        self._encoder_on_device = True

        # Set freeze state
        if self.encoder_freeze:
            self._freeze_encoder()
            self.logger.info("Encoder frozen")
        else:
            self.logger.info(f"Encoder trainable with lr={self.encoder_lr or 'default'}")

        # Load incident metadata if provided
        self.incident_metadata_path = cfg.get('TEST', {}).get('INCIDENT_METADATA_PATH')
        self.incident_slots = self._load_incident_metadata()

        # TTA configuration (disabled by default)
        self._init_tta(cfg)

        self.logger.info(f"Encoder type: {encoder_cfg.get('type')}")
        self.logger.info(f"Encoder source: {self.encoder_source}")
        self.logger.info(f"Encoder d_model: {self.encoder.d_model}")

    def _load_pretrained_encoder(self, ckpt_path: str):
        """Load pretrained encoder weights."""
        # Resolve glob pattern
        if '*' in ckpt_path:
            matches = glob.glob(ckpt_path)
            if not matches:
                raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")
            ckpt_path = sorted(matches)[-1]
            print(f"Resolved checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Delegate weight extraction and loading to encoder
        # Each encoder knows its own weight mapping
        missing, unexpected = self.encoder.load_pretrained_weights(state_dict, strict=False)

        if missing:
            print(f"Missing keys (will be randomly initialized): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys (ignored): {len(unexpected)}")

        # Count actually loaded parameters
        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_count = len(encoder_keys) - len(missing)
        print(f"Loaded {loaded_count}/{len(encoder_keys)} encoder parameters")

        if loaded_count == 0:
            raise RuntimeError(
                f"Failed to load any encoder parameters from checkpoint. "
                f"Check that checkpoint format matches encoder architecture."
            )

    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    # ==================== TTA (Test-Time Adaptation) ====================

    def _extract_prediction(self, model_output) -> torch.Tensor:
        """Extract prediction tensor from model output. Assumes dict with 'prediction' key."""
        return model_output['prediction']

    def _init_tta(self, cfg: Dict):
        """Initialize TTA configuration."""
        tta_cfg = cfg.get('TTA', {})

        self.tta_enabled = tta_cfg.get('ENABLED', False)
        self.tta_mode = tta_cfg.get('MODE', 'ensemble')

        # Ensemble settings
        self.tta_ensemble_runs = tta_cfg.get('ENSEMBLE_RUNS', 5)
        self.tta_ensemble_dropout = tta_cfg.get('ENSEMBLE_DROPOUT', 0.1)

        # Adaptation settings
        self.tta_adapt_steps = tta_cfg.get('ADAPT_STEPS', 1)
        self.tta_adapt_lr = tta_cfg.get('ADAPT_LR', 1e-4)
        self.tta_mask_ratio = tta_cfg.get('MASK_RATIO', 0.15)
        self.tta_adapt_layers = tta_cfg.get('ADAPT_LAYERS', 'all')

        # Create dropout layer for ensemble
        if self.tta_enabled:
            self.tta_dropout = nn.Dropout(p=self.tta_ensemble_dropout)
            self.logger.info(f"TTA enabled: mode={self.tta_mode}")
            if self.tta_mode in ['ensemble', 'both']:
                self.logger.info(f"  Ensemble: runs={self.tta_ensemble_runs}, dropout={self.tta_ensemble_dropout}")
            if self.tta_mode in ['adaptation', 'both']:
                self.logger.info(f"  Adaptation: steps={self.tta_adapt_steps}, lr={self.tta_adapt_lr}")

    def _tta_forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                     future_data_4_dec: torch.Tensor) -> torch.Tensor:
        """TTA forward pass dispatcher."""
        if self.tta_mode == 'ensemble':
            return self._tta_ensemble(history_data, future_data_4_dec)
        elif self.tta_mode == 'adaptation':
            return self._tta_adaptation(history_data, future_data_4_dec)
        elif self.tta_mode == 'both':
            return self._tta_both(history_data, future_data_4_dec)
        else:
            # Fall back to standard forward
            return self._standard_forward(history_data, future_data_4_dec)

    def _standard_forward(self, history_data: torch.Tensor,
                          future_data_4_dec: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (no TTA)."""
        with torch.no_grad():
            if self.output_reliability:
                result = self.encoder.encode(history_data, return_reliability=True)
                encoded, reliability = result if isinstance(result, tuple) else (result, None)
            else:
                encoded = self.encoder.encode(history_data)
                reliability = None
            encoded = self._add_tod_dow_if_needed(encoded, history_data)
            if reliability is not None:
                encoded = torch.cat([encoded, reliability], dim=-1)
            if self.downstream_features is not None:
                encoded = encoded[..., self.downstream_features]
            pred = self.model(
                history_data=encoded,
                future_data=future_data_4_dec,
                batch_seen=None, epoch=None, train=False
            )
            return self._extract_prediction(pred)

    def _tta_ensemble(self, history_data: torch.Tensor,
                      future_data_4_dec: torch.Tensor) -> torch.Tensor:
        """Ensemble TTA: multiple forward passes with dropout."""
        predictions = []

        for _ in range(self.tta_ensemble_runs):
            with torch.no_grad():
                encoded = self.encoder.encode(history_data)
                # Apply dropout for diversity
                self.tta_dropout.train()
                encoded = self.tta_dropout(encoded)
                encoded = self._add_tod_dow_if_needed(encoded, history_data)

                pred = self.model(
                    history_data=encoded,
                    future_data=future_data_4_dec,
                    batch_seen=None, epoch=None, train=False
                )
                predictions.append(self._extract_prediction(pred))

        return torch.stack(predictions, dim=0).mean(dim=0)

    def _tta_adaptation(self, history_data: torch.Tensor,
                        future_data_4_dec: torch.Tensor) -> torch.Tensor:
        """Adaptation TTA: adapt downstream model with prediction consistency loss.

        TTA assumes encoder is frozen - adapts the forecasting model instead.
        """
        adapt_params = [p for p in self.model.parameters() if p.requires_grad]
        if not adapt_params:
            return self._standard_forward(history_data, future_data_4_dec)

        saved_state = copy.deepcopy(self.model.state_dict())
        tta_optimizer = torch.optim.SGD(adapt_params, lr=self.tta_adapt_lr, momentum=0.9)

        for _ in range(self.tta_adapt_steps):
            self.model.train()
            with torch.no_grad():
                encoded = self.encoder.encode(history_data)
                encoded = self._add_tod_dow_if_needed(encoded, history_data)
                masked_data, _ = self._tta_random_mask(history_data)
                encoded_masked = self.encoder.encode(masked_data)
                encoded_masked = self._add_tod_dow_if_needed(encoded_masked, masked_data)

            # Enable gradients for adaptation (test loop disables them)
            with torch.enable_grad():
                # Prediction consistency: original and masked input should give similar predictions
                pred_orig = self._extract_prediction(
                    self.model(history_data=encoded, future_data=future_data_4_dec, batch_seen=None, epoch=None, train=True))
                with torch.no_grad():
                    pred_masked = self._extract_prediction(
                        self.model(history_data=encoded_masked, future_data=future_data_4_dec, batch_seen=None, epoch=None, train=False))

                loss = F.mse_loss(pred_orig, pred_masked.detach())
                tta_optimizer.zero_grad()
                loss.backward()
                tta_optimizer.step()

        # Predict with adapted model
        self.model.eval()
        with torch.no_grad():
            encoded = self.encoder.encode(history_data)
            encoded = self._add_tod_dow_if_needed(encoded, history_data)
            pred = self._extract_prediction(
                self.model(history_data=encoded, future_data=future_data_4_dec, batch_seen=None, epoch=None, train=False))

        self.model.load_state_dict(saved_state)
        return pred

    def _tta_both(self, history_data: torch.Tensor,
                  future_data_4_dec: torch.Tensor) -> torch.Tensor:
        """Combined TTA: adaptation then ensemble."""
        # Adapt model (state is restored after)
        adapt_params = [p for p in self.model.parameters() if p.requires_grad]
        if adapt_params:
            tta_optimizer = torch.optim.SGD(adapt_params, lr=self.tta_adapt_lr, momentum=0.9)
            for _ in range(self.tta_adapt_steps):
                self.model.train()
                with torch.no_grad():
                    encoded = self.encoder.encode(history_data)
                    encoded = self._add_tod_dow_if_needed(encoded, history_data)
                    masked_data, _ = self._tta_random_mask(history_data)
                    encoded_masked = self.encoder.encode(masked_data)
                    encoded_masked = self._add_tod_dow_if_needed(encoded_masked, masked_data)

                # Enable gradients for adaptation (test loop disables them)
                with torch.enable_grad():
                    pred_orig = self._extract_prediction(
                        self.model(history_data=encoded, future_data=future_data_4_dec, batch_seen=None, epoch=None, train=True))
                    with torch.no_grad():
                        pred_masked = self._extract_prediction(
                            self.model(history_data=encoded_masked, future_data=future_data_4_dec, batch_seen=None, epoch=None, train=False))

                    loss = F.mse_loss(pred_orig, pred_masked.detach())
                    tta_optimizer.zero_grad()
                    loss.backward()
                    tta_optimizer.step()

        # Ensemble with adapted model (don't restore - use adapted weights for ensemble)
        return self._tta_ensemble(history_data, future_data_4_dec)

    def _tta_random_mask(self, x: torch.Tensor) -> tuple:
        """Random node masking for TTA self-supervision."""
        B, T, N, C = x.shape
        mask = torch.rand(B, 1, N, 1, device=x.device) < self.tta_mask_ratio
        masked_x = x.clone()
        masked_x[mask.expand(B, T, N, C)] = 0
        return masked_x, mask

    def _add_tod_dow_if_needed(self, encoded: torch.Tensor,
                                history_data: torch.Tensor) -> torch.Tensor:
        """Add time-of-day, day-of-week, and pass-through features if configured."""
        parts = [encoded]
        if self.include_tod_dow:
            placeholder = encoded[..., 0:1]
            tod = history_data[..., self.tod_idx:self.tod_idx+1]
            dow = history_data[..., self.dow_idx:self.dow_idx+1]
            encoded_rest = encoded[..., 1:]
            parts = [placeholder, tod, dow, encoded_rest]
        if self.pass_through_indices:
            pass_channels = history_data[..., self.pass_through_indices]
            parts.append(pass_channels)
        encoded = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return encoded

    def build_optim(self, optim_cfg: Dict, model: nn.Module):
        """Build optimizer with optional discriminative learning rates."""
        optim_type = optim_cfg.get('TYPE', 'Adam')
        optim_param = optim_cfg.get('PARAM', {}).copy()
        base_lr = optim_param.pop('lr', 1e-3)

        # Parameter groups
        param_groups = [
            {
                'params': model.parameters(),
                'lr': base_lr,
                'name': 'downstream'
            },
        ]

        # Add encoder parameters if not frozen
        if not self.encoder_freeze:
            encoder_lr = self.encoder_lr if self.encoder_lr else base_lr
            param_groups.append({
                'params': self.encoder.parameters(),
                'lr': encoder_lr,
                'name': 'encoder'
            })

        optim_cls = getattr(torch.optim, optim_type)
        optimizer = optim_cls(param_groups, **optim_param)

        print(f"Optimizer: {optim_type}")
        for pg in param_groups:
            print(f"  {pg['name']}: lr={pg['lr']}")

        return optimizer

    def forward(
        self,
        data: Dict,
        epoch: Optional[int] = None,
        iter_num: Optional[int] = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """Forward pass with encoder transformation and optional TTA."""
        # Extract target mask before preprocessing (mask channel unaffected by scaler)
        target_mask = None
        if self.mask_channel_idx is not None:
            target_mask = data['target'][:, :, :, self.mask_channel_idx:self.mask_channel_idx + 1].clone()

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

        # Use TTA at test time if enabled
        if not train and self.tta_enabled:
            prediction = self._tta_forward(history_data, future_data, future_data_4_dec)
            model_return = {
                'prediction': prediction,
                'inputs': self.select_target_features(history_data),
                'target': self.select_target_features(future_data),
            }
            model_return = self.postprocessing(model_return)
            return model_return

        # Standard forward pass
        # Set encoder mode
        if train and not self.encoder_freeze:
            self.encoder.train()
        else:
            self.encoder.eval()

        # Encode with unified interface
        encode_kwargs = {'return_reliability': True} if self.output_reliability else {}
        if self.encoder_freeze:
            with torch.no_grad():
                encode_result = self.encoder.encode(history_data, **encode_kwargs)
        else:
            encode_result = self.encoder.encode(history_data, **encode_kwargs)

        if self.output_reliability and isinstance(encode_result, tuple):
            encoded_history, reliability = encode_result
        else:
            encoded_history = encode_result
            reliability = None

        # Optionally include tod/dow and pass-through channels
        encoded_history = self._add_tod_dow_if_needed(encoded_history, history_data)

        # Append reliability as last channel (before downstream feature selection)
        if reliability is not None:
            encoded_history = torch.cat([encoded_history, reliability], dim=-1)

        # Select downstream features (e.g., encoder outputs 5ch but STGCN needs only flow)
        if self.downstream_features is not None:
            encoded_history = encoded_history[..., self.downstream_features]

        # Forward through downstream model
        model_return = self.model(
            history_data=encoded_history,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        # Wrap tensor output in dict (some models return tensor directly)
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        model_return.setdefault('inputs', self.select_target_features(history_data))
        model_return.setdefault('target', self.select_target_features(future_data))

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"Output shape mismatch. Expected [B={batch_size}, L={length}, N={num_nodes}, C], got {model_return['prediction'].shape}"

        model_return = self.postprocessing(model_return)

        # Add target mask for mask-aware loss (after postprocessing to stay on correct device)
        if target_mask is not None:
            model_return['target_mask'] = target_mask.to(model_return['prediction'].device)

        return model_return

    # ==================== Checkpoint Save/Load ====================

    def _build_ckpt_dict(self, epoch: int) -> dict:
        """Build checkpoint dictionary with encoder state.

        Extends base checkpoint format with encoder-specific data.
        Used by both save_model and save_best_model.
        """
        return {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_config': self.encoder.get_config(),
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': getattr(self, 'best_metrics', {})
        }

    def save_model(self, epoch: int):
        """Save checkpoint with encoder state dict."""
        ckpt_dict = self._build_ckpt_dict(epoch)

        # Backup last checkpoint
        last_ckpt_path = self.get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # Save current checkpoint
        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        """Save best model with encoder state dict.

        Overrides base class to include encoder_state_dict.
        """
        metric = self.meter_pool.get_value(metric_name)
        best_metric = self.best_metrics.get(metric_name)

        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            ckpt_dict = self._build_ckpt_dict(epoch)

            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                f'{self.model_name}_best_{metric_name.replace("/", "_")}.pt'
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience  # reset patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load model and encoder state dicts from checkpoint."""
        checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)

        # Load model weights
        self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)

        # Load encoder weights if separately saved
        if 'encoder_state_dict' in checkpoint_dict:
            self.encoder.load_state_dict(checkpoint_dict['encoder_state_dict'], strict=strict)
            self.logger.info("Loaded encoder weights from checkpoint")

    # ==================== Incident Evaluation ====================

    def _load_incident_metadata(self) -> Optional[Dict]:
        """Load incident metadata for evaluation."""
        if not self.incident_metadata_path:
            return None
        if not os.path.exists(self.incident_metadata_path):
            return None

        incident_df = pd.read_csv(self.incident_metadata_path)
        incident_data = {}

        for _, row in incident_df.iterrows():
            incident_slot = int(row['input_start_slot']) + INCIDENT_SLOT_OFFSET
            incident_type = row['incident_type']

            if incident_type not in incident_data:
                incident_data[incident_type] = set()
            incident_data[incident_type].add(incident_slot)

        self.logger.info(f"Loaded incident metadata from {self.incident_metadata_path}")
        return incident_data

    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test with incident-specific evaluation."""
        self.encoder.eval()
        results = super().test(train_epoch, save_metrics, save_results)

        if self.incident_slots is not None:
            incident_metrics = self._evaluate_incidents()
            self._save_incident_metrics(incident_metrics)

        return results

    def _get_test_data_start_index(self) -> int:
        """Calculate test data start index."""
        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        valid_len = int(total_len * dataset.train_val_test_ratio[1])
        test_len = int(total_len * dataset.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len
        offset = dataset.input_len - 1 if dataset.overlap else 0
        return train_len + valid_len - offset

    def _get_test_len(self) -> int:
        """Calculate test data length."""
        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        return int(total_len * dataset.train_val_test_ratio[2])

    def _evaluate_incidents(self) -> Optional[Dict]:
        """Evaluate on incident time slots."""
        if not self.incident_slots:
            return None

        test_start_idx = self._get_test_data_start_index()
        test_end_idx = test_start_idx + self._get_test_len()

        # Get incident indices in test set
        incident_indices_by_type = {}
        for incident_type, slots in self.incident_slots.items():
            test_slots = {s for s in slots if test_start_idx <= s < test_end_idx}
            if test_slots:
                incident_indices_by_type[incident_type] = {s - test_start_idx for s in test_slots}

        if not incident_indices_by_type:
            return None

        # Collect predictions
        all_predictions, all_targets, all_indices = [], [], []
        for data in tqdm(self.test_data_loader, desc="Incident evaluation"):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()

            all_predictions.append(forward_return['prediction'])
            all_targets.append(forward_return['target'])
            all_indices.append(data['index'])

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        # Evaluate
        def evaluate_subset(indices, prefix):
            if not indices:
                return {}

            mask = torch.tensor([idx.item() in indices for idx in all_indices], dtype=torch.bool)
            if not mask.any():
                return {}

            pred, target = all_predictions[mask], all_targets[mask]
            returns = {'prediction': pred, 'target': target}

            result = {f'{prefix}_overall': {}}
            for name, func in self.metrics.items():
                val = self.metric_forward(func, returns)
                result[f'{prefix}_overall'][name] = val.item()

            for h in self.evaluation_horizons:
                horizon_returns = {'prediction': pred[:, h], 'target': target[:, h]}
                horizon_metrics = {}
                for name, func in self.metrics.items():
                    val = self.metric_forward(func, horizon_returns)
                    horizon_metrics[name] = val.item()
                result[f'{prefix}_horizon_{h+1}'] = horizon_metrics

            return result

        all_incident_metrics = {}

        # Overall incident vs normal
        all_incident_indices = set()
        for indices in incident_indices_by_type.values():
            all_incident_indices.update(indices)

        if all_incident_indices:
            all_incident_metrics.update(evaluate_subset(all_incident_indices, 'incident'))
            normal_indices = set(range(len(all_indices))) - all_incident_indices
            all_incident_metrics.update(evaluate_subset(normal_indices, 'normal'))

        # Per incident type
        for incident_type, indices in incident_indices_by_type.items():
            all_incident_metrics.update(evaluate_subset(indices, f'incident_{incident_type}'))

        return all_incident_metrics

    def _save_incident_metrics(self, incident_metrics: Optional[Dict]):
        """Save incident metrics."""
        if not incident_metrics:
            return

        save_path = os.path.join(self.ckpt_save_dir, 'test_incident_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(incident_metrics, f, indent=4)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("Incident vs Normal Performance:")
        self.logger.info("=" * 50)

        for key in ['incident_overall', 'normal_overall']:
            if key in incident_metrics:
                mae = incident_metrics[key]['MAE']
                self.logger.info(f"  {key}: MAE = {mae:.4f}")
