"""
Combined Runner: Pre-trained Encoder + Incident-Aware Evaluation

Combines:
- PretrainedEncoderRunner: Uses frozen encoder to transform input
- IncidentAwareRunner: Evaluates performance on incident cases
"""
import os
import json
from typing import Dict, Optional

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from .pretrained_encoder_runner import PretrainedEncoderRunner


class PretrainedIncidentRunner(PretrainedEncoderRunner):
    """
    Runner that:
    1. Applies pre-trained encoder to transform input data
    2. Supports incident-specific evaluation during testing
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Load incident metadata if provided
        self.incident_metadata_path = cfg.get('TEST', {}).get('INCIDENT_METADATA_PATH', None)
        self.incident_slots = self._load_incident_metadata()

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
            # Incident moment is at end of input window (input_start_slot + 11)
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

        # Get test data range
        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        test_len = int(total_len * dataset.train_val_test_ratio[2])
        test_end_idx = test_start_idx + test_len

        # Collect incident indices by type (only those in test range)
        incident_indices_by_type = {}
        for incident_type, incident_slots in self.incident_slots.items():
            test_incident_slots = {slot for slot in incident_slots if test_start_idx <= slot < test_end_idx}

            if test_incident_slots:
                relative_indices = {slot - test_start_idx for slot in test_incident_slots}
                incident_indices_by_type[incident_type] = relative_indices

        if not incident_indices_by_type:
            return None

        # Run prediction for all test data
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

        # Evaluate incident performance
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

            # Compute metrics using metric_forward
            result_metrics = {}
            result_metrics[f'{prefix}_overall'] = {}
            for metric_name, metric_func in self.metrics.items():
                metric_val = self.metric_forward(metric_func, returns)
                result_metrics[f'{prefix}_overall'][metric_name] = metric_val.item()

            # Per-horizon metrics
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

        # All incidents combined
        all_incident_indices = set()
        for indices in incident_indices_by_type.values():
            all_incident_indices.update(indices)

        if all_incident_indices:
            overall_metrics = evaluate_incident_performance(all_incident_indices, 'incident')
            all_incident_metrics.update(overall_metrics)

            # Non-incident (normal) cases
            all_test_indices = set(range(len(all_indices)))
            normal_indices = all_test_indices - all_incident_indices
            normal_metrics = evaluate_incident_performance(normal_indices, 'normal')
            all_incident_metrics.update(normal_metrics)

        # Per incident type
        for incident_type, indices in incident_indices_by_type.items():
            type_metrics = evaluate_incident_performance(indices, f'incident_{incident_type}')
            all_incident_metrics.update(type_metrics)

        return all_incident_metrics

    def _save_incident_metrics(self, incident_metrics: Optional[Dict]):
        """Save incident metrics to JSON file."""
        if incident_metrics is None:
            return

        save_path = os.path.join(self.ckpt_save_dir, 'test_incident_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(incident_metrics, f, indent=4)

        # Print summary
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Incident vs Normal Performance:")
        self.logger.info("=" * 50)

        for key in ['incident_overall', 'normal_overall']:
            if key in incident_metrics:
                mae = incident_metrics[key].get('MAE', 'N/A')
                self.logger.info(f"  {key}: MAE = {mae:.4f}" if isinstance(mae, float) else f"  {key}: MAE = {mae}")

        # Horizon-wise comparison
        for horizon in [3, 6, 12]:
            inc_key = f'incident_horizon_{horizon}'
            norm_key = f'normal_horizon_{horizon}'
            if inc_key in incident_metrics and norm_key in incident_metrics:
                inc_mae = incident_metrics[inc_key].get('MAE', 0)
                norm_mae = incident_metrics[norm_key].get('MAE', 0)
                self.logger.info(f"  Horizon {horizon}: Incident MAE={inc_mae:.4f}, Normal MAE={norm_mae:.4f}")
