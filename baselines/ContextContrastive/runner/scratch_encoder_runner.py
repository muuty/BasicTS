"""
Scratch Encoder Runner: Train encoder from scratch (no pre-training).

This runner trains the encoder jointly with the downstream model,
allowing us to test if the encoder architecture itself provides value
without contrastive pre-training.
"""
import os
import json
import pickle
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from basicts.runners import SimpleTimeSeriesForecastingRunner


class ScratchEncoderRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner that trains encoder from scratch along with downstream model.

    Config requirements:
        CFG.ENCODER: dict with keys:
            - type: 'temporal' or 'spatiotemporal'
            - d_model: encoder output dimension
            - input_dim: original input dimension
            - temporal_layers, temporal_heads: temporal encoder config
            - spatial_layers, spatial_heads, k_neighbors: spatial encoder config (if spatiotemporal)
            - encoder_lr: learning rate for encoder (optional, defaults to main lr)
            - include_tod_dow: if True, include tod/dow in output
        CFG.DATASET.PARAM.adj_path: path to adjacency matrix (for spatiotemporal)
    """

    def __init__(self, cfg: Dict):
        # Store encoder config before calling super().__init__
        encoder_cfg = cfg.get('ENCODER', {})
        if not encoder_cfg:
            raise ValueError("ENCODER config is required")

        self.encoder_cfg = encoder_cfg
        self.encoder_type = encoder_cfg.get('type', 'temporal')
        self.encoder_lr = encoder_cfg.get('encoder_lr', None)
        self.include_tod_dow = encoder_cfg.get('include_tod_dow', False)

        # Build encoder
        self.encoder = self._build_encoder(encoder_cfg)
        self._encoder_on_device = False

        # Load adjacency matrix if needed
        self.adj = None
        if self.encoder_type == 'spatiotemporal':
            adj_path = encoder_cfg.get('adj_path', None)
            if adj_path is None:
                # Try default path
                dataset_name = cfg.get('DATASET', {}).get('NAME', '')
                adj_path = f'datasets/{dataset_name}/adj_mx.pkl'
            self.adj = self._load_adjacency(adj_path)

        # Call parent init
        super().__init__(cfg)

        # Load incident metadata if provided
        self.incident_metadata_path = cfg.get('TEST', {}).get('INCIDENT_METADATA_PATH', None)
        self.incident_slots = self._load_incident_metadata()

        self.logger.info(f"Encoder type: {self.encoder_type}")
        self.logger.info(f"Encoder d_model: {encoder_cfg.get('d_model')}")
        if self.adj is not None:
            self.logger.info(f"Adjacency matrix loaded: {self.adj.shape}")

    def _build_encoder(self, encoder_cfg: Dict) -> nn.Module:
        """Build encoder from config."""
        encoder_type = encoder_cfg.get('type', 'temporal')

        if encoder_type == 'temporal':
            from ..arch import ContextAwareEncoder
            return ContextAwareEncoder(
                input_dim=encoder_cfg.get('input_dim', 3),
                d_model=encoder_cfg.get('d_model', 64),
                num_layers=encoder_cfg.get('temporal_layers', 2),
                nhead=encoder_cfg.get('temporal_heads', 4),
                dropout=encoder_cfg.get('dropout', 0.1),
            )
        elif encoder_type == 'spatiotemporal':
            from ..arch import SpatioTemporalEncoder
            return SpatioTemporalEncoder(
                input_dim=encoder_cfg.get('input_dim', 3),
                d_model=encoder_cfg.get('d_model', 64),
                temporal_layers=encoder_cfg.get('temporal_layers', 2),
                temporal_heads=encoder_cfg.get('temporal_heads', 4),
                spatial_layers=encoder_cfg.get('spatial_layers', 1),
                spatial_heads=encoder_cfg.get('spatial_heads', 4),
                k_neighbors=encoder_cfg.get('k_neighbors', 10),
                dropout=encoder_cfg.get('dropout', 0.1),
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _load_adjacency(self, adj_path: str) -> Optional[torch.Tensor]:
        """Load adjacency matrix from file."""
        if not os.path.exists(adj_path):
            self.logger.warning(f"Adjacency file not found: {adj_path}")
            return None

        with open(adj_path, 'rb') as f:
            adj = pickle.load(f)

        # Handle different formats
        if isinstance(adj, (list, tuple)):
            adj = adj[-1]  # Usually the last element is the adjacency matrix

        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float()

        print(f"Loaded adjacency matrix: {adj.shape}")
        return adj

    def build_optim(self, optim_cfg: Dict, model: nn.Module):
        """Build optimizer with optional separate LR for encoder."""
        optim_type = optim_cfg.get('TYPE', 'Adam')
        optim_param = optim_cfg.get('PARAM', {}).copy()
        base_lr = optim_param.pop('lr', 1e-3)

        # Create parameter groups
        param_groups = [
            {
                'params': model.parameters(),
                'lr': base_lr,
                'name': 'downstream'
            },
            {
                'params': self.encoder.parameters(),
                'lr': self.encoder_lr if self.encoder_lr else base_lr,
                'name': 'encoder'
            },
        ]

        # Build optimizer
        optim_cls = getattr(torch.optim, optim_type)
        optimizer = optim_cls(param_groups, **optim_param)

        print(f"Optimizer: {optim_type}")
        for pg in param_groups:
            print(f"  {pg['name']}: lr={pg['lr']}")

        return optimizer

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """Forward pass with encoder transformation."""
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Move encoder to device if needed
        if not self._encoder_on_device:
            self.encoder = self.encoder.to(history_data.device)
            if self.adj is not None:
                self.adj = self.adj.to(history_data.device)
            self._encoder_on_device = True

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Set encoder mode
        if train:
            self.encoder.train()
        else:
            self.encoder.eval()

        # Encode history_data
        if self.encoder_type == 'spatiotemporal' and self.adj is not None:
            encoded_history = self.encoder(history_data, self.adj)
        else:
            encoded_history = self.encoder(history_data)

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
        self.encoder.eval()
        results = super().test(train_epoch, save_metrics, save_results)

        if self.incident_slots is not None:
            incident_metrics = self._evaluate_incidents()
            self._save_incident_metrics(incident_metrics)

        return results

    def _evaluate_incidents(self) -> Optional[Dict]:
        """Evaluate model performance on incident time slots."""
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

        def evaluate_subset(indices, prefix):
            if not indices:
                return {}

            mask = torch.tensor([idx.item() in indices for idx in all_indices], dtype=torch.bool)
            if not mask.any():
                return {}

            prediction = all_predictions[mask]
            target = all_targets[mask]
            returns = {'prediction': prediction, 'target': target}

            result_metrics = {f'{prefix}_overall': {}}
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
            all_incident_metrics.update(evaluate_subset(all_incident_indices, 'incident'))
            all_test_indices = set(range(len(all_indices)))
            normal_indices = all_test_indices - all_incident_indices
            all_incident_metrics.update(evaluate_subset(normal_indices, 'normal'))

        for incident_type, indices in incident_indices_by_type.items():
            all_incident_metrics.update(evaluate_subset(indices, f'incident_{incident_type}'))

        return all_incident_metrics

    # ==================== Checkpoint Save/Load with Encoder ====================

    def save_model(self, epoch: int):
        """Save checkpoint with both model and encoder state dicts.

        This ensures evaluation can reproduce training results by saving
        the encoder weights alongside the downstream model.
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        from easytorch.utils import save_ckpt

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),  # Save encoder weights
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': getattr(self, 'best_metrics', None)
        }
        save_ckpt(ckpt_dict, self.ckpt_save_dir, epoch, self.model_name,
                  self.num_ckpts, self.target_metrics, self.best_metrics, self.logger)

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

            # Load encoder weights if available
            if 'encoder_state_dict' in checkpoint_dict:
                self.encoder.load_state_dict(checkpoint_dict['encoder_state_dict'], strict=strict)
                self.logger.info("Loaded encoder weights from checkpoint")
            else:
                self.logger.warning("No encoder_state_dict in checkpoint - encoder initialized from scratch")

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
