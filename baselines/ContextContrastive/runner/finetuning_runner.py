"""
Fine-tuning Runner: Supports unfreezing the pre-trained encoder for end-to-end training.

Features:
- Discriminative learning rates (lower LR for encoder, higher for downstream)
- Optional gradual unfreezing after N epochs
- Incident-aware evaluation support
"""
import os
import json
import glob
from typing import Dict, Optional

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from basicts.runners import SimpleTimeSeriesForecastingRunner
from ..arch import ContextAwareEncoder, SpatioTemporalEncoder
from baselines.TS2Vec.arch import TS2VecEncoder


class FineTuningRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner that fine-tunes the pre-trained encoder along with the downstream model.

    Config requirements:
        CFG.PRETRAINED_ENCODER: dict with keys:
            - ckpt_path: path to pre-trained checkpoint
            - d_model: encoder output dimension
            - input_dim: original input dimension
            - num_layers, nhead, dropout: encoder architecture
            - freeze: if False, encoder will be trained (default: False)
            - encoder_lr: learning rate for encoder (default: 1e-5)
            - unfreeze_after: epoch to start unfreezing (default: 0)
            - include_tod_dow: if True, include tod/dow in output
    """

    def __init__(self, cfg: Dict):
        # Store encoder config before calling super().__init__
        encoder_cfg = cfg.get('PRETRAINED_ENCODER', {})
        if not encoder_cfg:
            raise ValueError("PRETRAINED_ENCODER config is required")

        self.encoder_cfg = encoder_cfg
        self.freeze_encoder = encoder_cfg.get('freeze', False)
        self.encoder_lr = encoder_cfg.get('encoder_lr', 1e-5)
        self.unfreeze_after = encoder_cfg.get('unfreeze_after', 0)
        self.include_tod_dow = encoder_cfg.get('include_tod_dow', False)
        self._encoder_frozen = True  # Initialize state

        # Build encoder before super().__init__() (needed for optimizer)
        encoder_result = self._build_encoder(encoder_cfg)
        if isinstance(encoder_result, tuple):
            self.encoder, self.adj_matrix = encoder_result
        else:
            self.encoder, self.adj_matrix = encoder_result, None
        self._load_encoder_weights(encoder_cfg.get('ckpt_path'))

        # Call parent init (this will call build_model -> build_optim)
        super().__init__(cfg)

        # Move encoder to same device as model (critical for optimizer)
        device = next(self.model.parameters()).device
        self.encoder = self.encoder.to(device)
        if self.adj_matrix is not None:
            self.adj_matrix = self.adj_matrix.to(device)

        # Set initial freeze state
        if self.freeze_encoder or self.unfreeze_after > 0:
            self._set_encoder_frozen(True)
            self.logger.info(f"Encoder initially frozen (unfreeze_after={self.unfreeze_after})")
        else:
            self._set_encoder_frozen(False)
            self.logger.info(f"Encoder unfrozen with lr={self.encoder_lr}")

        # Load incident metadata if provided
        self.incident_metadata_path = cfg.get('TEST', {}).get('INCIDENT_METADATA_PATH', None)
        self.incident_slots = self._load_incident_metadata()

        self.logger.info(f"Loaded pre-trained encoder from {encoder_cfg.get('ckpt_path')}")
        self.logger.info(f"Encoder output dimension: {encoder_cfg.get('d_model')}")

    def _build_encoder(self, encoder_cfg: Dict):
        """Build encoder from config. Supports multiple encoder types."""
        encoder_type = encoder_cfg.get('encoder_type', 'ContextAwareEncoder')

        if encoder_type == 'SpatioTemporalEncoder':
            # GAT-based encoder with adjacency matrix
            import torch
            from basicts.utils import load_adj

            # Load adjacency matrix if path provided
            adj_matrix = encoder_cfg.get('adj_matrix', None)
            if adj_matrix is None and 'adj_path' in encoder_cfg:
                adj_mx, _ = load_adj(encoder_cfg['adj_path'], "doubletransition")
                adj_matrix = torch.Tensor(adj_mx[0])

            return SpatioTemporalEncoder(
                input_dim=encoder_cfg.get('input_dim', 3),
                d_model=encoder_cfg.get('d_model', 64),
                temporal_layers=encoder_cfg.get('temporal_layers', encoder_cfg.get('num_layers', 2)),
                temporal_heads=encoder_cfg.get('temporal_heads', encoder_cfg.get('nhead', 4)),
                spatial_layers=encoder_cfg.get('spatial_layers', 1),
                spatial_heads=encoder_cfg.get('spatial_heads', 4),
                k_neighbors=encoder_cfg.get('k_neighbors', 10),
                dropout=encoder_cfg.get('dropout', 0.1),
            ), adj_matrix
        elif encoder_type == 'TS2VecEncoder':
            # TS2Vec dilated conv encoder
            return TS2VecEncoder(
                input_dim=encoder_cfg.get('input_dim', 3),
                output_dim=encoder_cfg.get('d_model', 64),
                hidden_dim=encoder_cfg.get('hidden_dim', 64),
                depth=encoder_cfg.get('depth', 10),
                mask_mode='all_true',  # No masking during fine-tuning
                dropout=encoder_cfg.get('dropout', 0.1),
            ), None
        else:
            # Default: ContextAwareEncoder (temporal only)
            return ContextAwareEncoder(
                input_dim=encoder_cfg.get('input_dim', 3),
                d_model=encoder_cfg.get('d_model', 64),
                num_layers=encoder_cfg.get('num_layers', 2),
                nhead=encoder_cfg.get('nhead', 4),
                dropout=encoder_cfg.get('dropout', 0.1),
            ), None

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
            print(f"Resolved checkpoint path: {ckpt_path}")

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

        # Handle legacy key names (old format → new format)
        key_mappings = [
            ('temporal_input_proj', 'temporal_encoder.input_proj'),
            ('temporal_transformer', 'temporal_encoder.transformer'),
            ('spatial_gat_layers', 'spatial_encoder.gat_layers'),
            ('spatial_layer_norms', 'spatial_encoder.layer_norms'),
        ]

        remapped_state_dict = {}
        for key, value in encoder_state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mappings:
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            remapped_state_dict[new_key] = value

        self.encoder.load_state_dict(remapped_state_dict)
        print(f"Loaded {len(encoder_state_dict)} encoder parameters")

    def _set_encoder_frozen(self, frozen: bool):
        """Set encoder frozen state."""
        for param in self.encoder.parameters():
            param.requires_grad = not frozen
        if frozen:
            self.encoder.eval()
        else:
            self.encoder.train()
        self._encoder_frozen = frozen

    def build_optim(self, optim_cfg: Dict, model: nn.Module):
        """Build optimizer with discriminative learning rates."""
        # Get optimizer type and params
        optim_type = optim_cfg.get('TYPE', 'Adam')
        optim_param = optim_cfg.get('PARAM', {}).copy()
        base_lr = optim_param.pop('lr', 1e-3)

        # Create parameter groups
        param_groups = [
            # Downstream model parameters
            {
                'params': model.parameters(),
                'lr': base_lr,
                'name': 'downstream'
            },
        ]

        # Add encoder parameters if not frozen
        if not self.freeze_encoder:
            param_groups.append({
                'params': self.encoder.parameters(),
                'lr': self.encoder_lr,
                'name': 'encoder'
            })

        # Build optimizer
        optim_cls = getattr(torch.optim, optim_type)
        optimizer = optim_cls(param_groups, **optim_param)

        print(f"Optimizer: {optim_type}")
        for pg in param_groups:
            print(f"  {pg['name']}: lr={pg['lr']}")

        return optimizer

    def on_epoch_start(self, epoch: int):
        """Handle gradual unfreezing at epoch start."""
        super().on_epoch_start(epoch)

        # Check if we should unfreeze encoder
        if hasattr(self, '_encoder_frozen') and self._encoder_frozen:
            if epoch >= self.unfreeze_after:
                self._set_encoder_frozen(False)
                # Add encoder params to optimizer
                self.optim.add_param_group({
                    'params': self.encoder.parameters(),
                    'lr': self.encoder_lr,
                    'name': 'encoder'
                })
                self.logger.info(f"Epoch {epoch}: Unfreezing encoder with lr={self.encoder_lr}")

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """Forward pass with encoder transformation."""
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

        # Encode history_data
        # Use torch.no_grad() only if encoder is frozen
        # Pass adjacency matrix if available (for SpatioTemporalEncoder)
        if self._encoder_frozen:
            with torch.no_grad():
                encoded_history = self.encoder(history_data, self.adj_matrix)
        else:
            encoded_history = self.encoder(history_data, self.adj_matrix)

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

    # ==================== Incident Evaluation ====================

    def _load_incident_metadata(self) -> Optional[Dict]:
        """Load incident metadata and extract time slots by type."""
        if self.incident_metadata_path is None:
            return None

        if not os.path.exists(self.incident_metadata_path):
            self.logger.warning(f"Incident metadata file not found: {self.incident_metadata_path}")
            return None

        incident_df = pd.read_csv(self.incident_metadata_path)
        incident_data = {}

        for _, row in incident_df.iterrows():
            incident_slot = int(row['input_start_slot']) + 11
            incident_type = row['incident_type']

            if incident_type not in incident_data:
                incident_data[incident_type] = set()
            incident_data[incident_type].add(incident_slot)

        self.logger.info(f"Loaded incident time slots from {self.incident_metadata_path}")
        for incident_type, slots in incident_data.items():
            self.logger.info(f"  {incident_type}: {len(slots)} incidents")

        return incident_data

    def _get_test_data_start_index(self) -> int:
        """Calculate the start index of test data."""
        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        valid_len = int(total_len * dataset.train_val_test_ratio[1])
        test_len = int(total_len * dataset.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        offset = dataset.input_len - 1 if dataset.overlap else 0
        return train_len + valid_len - offset

    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process with incident-specific evaluation support."""
        # Set encoder to eval mode for testing
        self.encoder.eval()

        # Run normal test first
        results = super().test(train_epoch, save_metrics, save_results)

        # Run incident-specific evaluation if metadata is available
        if self.incident_slots is not None:
            incident_metrics = self._evaluate_incidents()
            self._save_incident_metrics(incident_metrics)

        return results

    def _evaluate_incidents(self) -> Optional[Dict]:
        """Evaluate model performance on incident time slots by type."""
        if self.incident_slots is None:
            return None

        test_start_idx = self._get_test_data_start_index()

        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        test_len = int(total_len * dataset.train_val_test_ratio[2])
        test_end_idx = test_start_idx + test_len

        incident_indices_by_type = {}
        for incident_type, incident_slots in self.incident_slots.items():
            test_incident_slots = {slot for slot in incident_slots if test_start_idx <= slot < test_end_idx}

            if test_incident_slots:
                relative_indices = {slot - test_start_idx for slot in test_incident_slots}
                incident_indices_by_type[incident_type] = relative_indices

        if not incident_indices_by_type:
            return None

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

        all_incident_metrics = {}

        def evaluate_incident_performance(incident_indices, prefix):
            if not incident_indices:
                return {}

            mask = torch.tensor([idx.item() in incident_indices for idx in all_indices], dtype=torch.bool)
            if not mask.any():
                return {}

            prediction = all_predictions[mask]
            target = all_targets[mask]

            returns = {'prediction': prediction, 'target': target}

            result_metrics = {}
            result_metrics[f'{prefix}_overall'] = {}
            for metric_name, metric_func in self.metrics.items():
                metric_val = self.metric_forward(metric_func, returns)
                result_metrics[f'{prefix}_overall'][metric_name] = metric_val.item()

            for i in self.evaluation_horizons:
                pred_h = prediction[:, i, :, :]
                target_h = target[:, i, :, :]
                horizon_returns = {'prediction': pred_h, 'target': target_h}
                horizon_metrics = {}
                for metric_name, metric_func in self.metrics.items():
                    metric_val = self.metric_forward(metric_func, horizon_returns)
                    horizon_metrics[metric_name] = metric_val.item()
                result_metrics[f'{prefix}_horizon_{i+1}'] = horizon_metrics

            return result_metrics

        all_incident_indices = set()
        for indices in incident_indices_by_type.values():
            all_incident_indices.update(indices)

        if all_incident_indices:
            overall_metrics = evaluate_incident_performance(all_incident_indices, 'incident')
            all_incident_metrics.update(overall_metrics)

            all_test_indices = set(range(len(all_indices)))
            normal_indices = all_test_indices - all_incident_indices
            normal_metrics = evaluate_incident_performance(normal_indices, 'normal')
            all_incident_metrics.update(normal_metrics)

        for incident_type, indices in incident_indices_by_type.items():
            type_metrics = evaluate_incident_performance(indices, f'incident_{incident_type}')
            all_incident_metrics.update(type_metrics)

        return all_incident_metrics

    # ==================== Checkpoint Save/Load with Encoder ====================

    def save_model(self, epoch: int):
        """Save checkpoint with both model and encoder state dicts.

        This ensures evaluation can reproduce training results by saving
        the fine-tuned encoder weights alongside the downstream model.
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        from easytorch.core.checkpoint import save_ckpt

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),  # Save encoder weights
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': getattr(self, 'best_metrics', None)
        }
        # Save epoch checkpoint
        ckpt_path = os.path.join(self.ckpt_save_dir, f'{self.model_name}_{epoch:03d}.pt')
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # Save best checkpoint if this is the best so far
        current_metric = self.meter_pool.get_value('val/MAE')
        best_metric = self.best_metrics.get('val_MAE') if self.best_metrics else None
        if best_metric is None or current_metric < best_metric:
            if self.best_metrics is None:
                self.best_metrics = {}
            self.best_metrics['val_MAE'] = current_metric
            ckpt_dict['best_metrics'] = self.best_metrics
            best_ckpt_path = os.path.join(self.ckpt_save_dir, f'{self.model_name}_best_val_MAE.pt')
            save_ckpt(ckpt_dict, best_ckpt_path, self.logger)

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load model and encoder state dicts from checkpoint.

        This ensures evaluation uses the same encoder weights as training.
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        from easytorch.core.checkpoint import load_ckpt

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)

            # Load model weights
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)

            # Load encoder weights if available (for checkpoints saved after this fix)
            if 'encoder_state_dict' in checkpoint_dict:
                self.encoder.load_state_dict(checkpoint_dict['encoder_state_dict'], strict=strict)
                self.logger.info("Loaded fine-tuned encoder weights from checkpoint")
            else:
                self.logger.warning("No encoder_state_dict in checkpoint - using pretrained weights")

        except (IndexError, OSError) as e:
            raise OSError('Ckpt file does not exist') from e

    def _save_incident_metrics(self, incident_metrics: Optional[Dict]):
        """Save incident metrics to JSON file."""
        if incident_metrics is None:
            return

        save_path = os.path.join(self.ckpt_save_dir, 'test_incident_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(incident_metrics, f, indent=4)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("Incident vs Normal Performance:")
        self.logger.info("=" * 50)

        for key in ['incident_overall', 'normal_overall']:
            if key in incident_metrics:
                mae = incident_metrics[key].get('MAE', 'N/A')
                self.logger.info(f"  {key}: MAE = {mae:.4f}" if isinstance(mae, float) else f"  {key}: MAE = {mae}")

        for horizon in [3, 6, 12]:
            inc_key = f'incident_horizon_{horizon}'
            norm_key = f'normal_horizon_{horizon}'
            if inc_key in incident_metrics and norm_key in incident_metrics:
                inc_mae = incident_metrics[inc_key].get('MAE', 0)
                norm_mae = incident_metrics[norm_key].get('MAE', 0)
                self.logger.info(f"  Horizon {horizon}: Incident MAE={inc_mae:.4f}, Normal MAE={norm_mae:.4f}")
