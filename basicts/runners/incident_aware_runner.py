import os
import json
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from easytorch.utils import master_only

from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings

from contrastive.contrastive_loss import MinimalAntiSmoothingLoss


class IncidentAwareRunner(SimpleTimeSeriesForecastingRunner):
    """
    Custom runner that supports incident-specific evaluation.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.incident_metadata_path = cfg.get('TEST', {}).get('INCIDENT_METADATA_PATH', None)
        self.incident_slots = self._load_incident_metadata() if self.incident_metadata_path else None

        self.has_contrastive_loss = cfg.get('CONTRASTIVE_LOSS', None) is not None
        if self.has_contrastive_loss:
            self.contrastive_loss = cfg['CONTRASTIVE_LOSS']
            self.cl_weight = cfg['CONTRASTIVE_LOSS_WEIGHT']
            self.register_epoch_meter('train/cl_loss', 'train', '{:.4f}')
            self.register_epoch_meter('train/pred_loss', 'train', '{:.4f}')

    def _load_incident_metadata(self) -> Optional[Dict]:
        """Load incident metadata and extract time slots by type."""
        if not os.path.exists(self.incident_metadata_path):
            print(f"Warning: Incident metadata file not found: {self.incident_metadata_path}")
            return None

        incident_df = pd.read_csv(self.incident_metadata_path)
        incident_data = {}

        for _, row in incident_df.iterrows():
            incident_slot = int(row['input_start_slot']) + 11
            incident_type = row['incident_type']

            if incident_type not in incident_data:
                incident_data[incident_type] = set()
            incident_data[incident_type].add(incident_slot)

        print(f"Loaded incident time slots by type from {self.incident_metadata_path}")
        for incident_type, slots in incident_data.items():
            print(f"  {incident_type}: {len(slots)} incidents")

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

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process with incident-specific evaluation.

        Extends parent's test loop to also collect sample indices,
        then reuses predictions for incident evaluation (no redundant forward pass).
        """

        prediction, target, inputs, indices = [], [], [], []

        for data in tqdm(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter('test/loss', loss.item())

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])
            indices.append(data['index'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)
        all_indices = torch.cat(indices, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        self.compute_evaluation_metrics(returns_all)

        if save_results:
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            metrics_results = self.compute_evaluation_metrics(returns_all)
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        # Incident evaluation: reuse predictions, no second forward pass
        if self.incident_slots is not None:
            incident_metrics = self._evaluate_incidents(returns_all, all_indices)
            self._save_incident_metrics(incident_metrics)

        return returns_all

    def _evaluate_incidents(self, returns_all: Dict, all_indices: torch.Tensor) -> Optional[Dict]:
        """Evaluate model performance on incident time slots by type.

        Reuses pre-computed predictions from test() instead of running
        a redundant forward pass. Uses batch CPU transfer instead of
        per-element .item() calls to avoid GPU-CPU sync bottleneck.
        """
        print("Evaluating on incident time slots by type...")

        test_start_idx = self._get_test_data_start_index()

        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        test_len = int(total_len * dataset.train_val_test_ratio[2])
        test_end_idx = test_start_idx + test_len

        incident_indices_by_type = {}
        all_incident_indices = set()

        for incident_type, incident_slots in self.incident_slots.items():
            test_incident_slots = {slot for slot in incident_slots if test_start_idx <= slot < test_end_idx}

            if test_incident_slots:
                relative_incident_indices = {slot - test_start_idx for slot in test_incident_slots}
                incident_indices_by_type[incident_type] = relative_incident_indices
                all_incident_indices.update(relative_incident_indices)
                print(f"  {incident_type}: {len(relative_incident_indices)} incidents in test range")

        if not incident_indices_by_type:
            print("No incident types found in test data range")
            return None

        print(f"  Total unique incidents: {len(all_incident_indices)}")

        all_predictions = returns_all['prediction']
        all_targets = returns_all['target']
        all_inputs = returns_all['inputs']

        # Single batch GPU→CPU transfer instead of per-element .item() calls
        indices_list = all_indices.cpu().tolist()

        print(f"Total test samples: {len(all_predictions)}")

        all_incident_metrics = {}

        # === 1. Non-incident metrics ===
        non_incident_mask = torch.tensor(
            [idx not in all_incident_indices for idx in indices_list],
            dtype=torch.bool
        )
        if non_incident_mask.any():
            print(f"Evaluating non-incident samples: {non_incident_mask.sum().item()}")
            non_incident_returns = {
                'prediction': all_predictions[non_incident_mask],
                'target': all_targets[non_incident_mask],
                'inputs': all_inputs[non_incident_mask]
            }
            non_incident_metrics = self.compute_evaluation_metrics(non_incident_returns)
            all_incident_metrics.update({f'non_incident_{k}': v for k, v in non_incident_metrics.items()})

        # === 2. All incidents combined metrics ===
        all_incident_mask = ~non_incident_mask
        if all_incident_mask.any():
            print(f"Evaluating all incident samples: {all_incident_mask.sum().item()}")
            all_incident_returns = {
                'prediction': all_predictions[all_incident_mask],
                'target': all_targets[all_incident_mask],
                'inputs': all_inputs[all_incident_mask]
            }
            combined_metrics = self.compute_evaluation_metrics(all_incident_returns)
            all_incident_metrics.update({f'all_incident_{k}': v for k, v in combined_metrics.items()})

        # === 3. Per incident type metrics ===
        for incident_type, incident_indices in incident_indices_by_type.items():
            print(f"Evaluating {incident_type} incidents...")

            incident_mask = torch.tensor(
                [idx in incident_indices for idx in indices_list],
                dtype=torch.bool
            )

            if not incident_mask.any():
                print(f"  No {incident_type} samples found in test data")
                continue

            print(f"  Evaluating on {incident_mask.sum().item()} {incident_type} samples")

            incident_returns = {
                'prediction': all_predictions[incident_mask],
                'target': all_targets[incident_mask],
                'inputs': all_inputs[incident_mask]
            }
            incident_metrics = self.compute_evaluation_metrics(incident_returns)
            all_incident_metrics.update({f'{incident_type}_{k}': v for k, v in incident_metrics.items()})

        return all_incident_metrics
    
    def _save_incident_metrics(self, incident_metrics: Dict):
        """Save incident metrics to JSON file."""
        if incident_metrics is None:
            print("No incident metrics to save")
            return
            
        save_path = os.path.join(self.ckpt_save_dir, 'test_incident_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(incident_metrics, f, indent=4)
        
        print(f"Incident metrics saved to: {save_path}")
        
        # Print summary
        # print("\nIncident Type Performance Summary:")
        incident_types = set()
        for key in incident_metrics.keys():
            if '_horizon_' in key:
                incident_type = key.split('_horizon_')[0]
                incident_types.add(incident_type)
        
        for incident_type in sorted(incident_types):
            # print(f"\n{incident_type}:"
            for horizon in [3, 6, 12]:
                mae_key = f'{incident_type}_horizon_{horizon}'
                if mae_key in incident_metrics:
                    mae_value = incident_metrics[mae_key]['MAE']
                    # print(f"  Horizon {horizon}: MAE = {mae_value:.4f}")

    def train_iters(self, epoch, iter_index, data):
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # 예측 손실
        pred_loss = self.metric_forward(self.loss, forward_return)
        if self.has_contrastive_loss:
            cl_loss = self.contrastive_loss(forward_return)
            total_loss = pred_loss + self.cl_weight * cl_loss
        else:
            total_loss = pred_loss

        # === Joint Training: Replay loss를 여기서 계산하여 합침 ===
        if self.has_experience_replay and epoch > 1 and self.replay.size() > 0:
            replay_loss = self.compute_replay_loss(epoch)
            if replay_loss is not None:
                total_loss = total_loss + self.replay_weight * replay_loss
                self.update_epoch_meter('train/replay_loss', replay_loss.item())

        # Replay 버퍼 업데이트 (loss 계산 후에)
        if self.has_experience_replay:
            self.replay.push_batch(data=data, forward_return=forward_return)

        self.update_epoch_meter('train/loss', total_loss.item())
        if self.has_contrastive_loss:
            self.update_epoch_meter('train/pred_loss', pred_loss.item())
            self.update_epoch_meter('train/cl_loss', self.cl_weight * cl_loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return total_loss

    def compute_replay_loss(self, epoch: int) -> torch.Tensor:
        """Replay loss만 계산하여 반환 (backward 없이)"""
        indices = self.replay.sample(self.replay.batch_size)
        
        if len(indices) == 0:
            return None
        
        dataset = self.train_data_loader.dataset
        samples = [dataset[idx] for idx in indices]
        
        batch = {
            'inputs': torch.stack([torch.from_numpy(s['inputs']) for s in samples]),
            'target': torch.stack([torch.from_numpy(s['target']) for s in samples]),
            'index': torch.tensor([s['index'] for s in samples])
        }
        
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        forward_return = self.forward(batch, epoch=epoch, iter_num=None, train=True)

        prediction = forward_return['prediction']
        target = forward_return['target']
        _, T, N, _ = prediction.shape
        replay_loss = (prediction - target).abs().mean(dim=3).sum() / (T * N) 
        return replay_loss  # backward 없이 loss만 반환

