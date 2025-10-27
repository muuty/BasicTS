import os
import json
from typing import Dict, Optional, Set

import pandas as pd
import torch
from tqdm import tqdm

from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings

from contrastive.contrastive_loss import MinimalAntiSmoothingLoss


class IncidentAwareRunner(SimpleTimeSeriesForecastingRunner):
    """
    Custom runner that supports incident-specific evaluation.
    """
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.incident_metadata_path = cfg['TEST']['INCIDENT_METADATA_PATH']
        self.incident_slots = self._load_incident_metadata()
        

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
            # 사고 순간은 input의 마지막 샘플 (input_start_slot + 11)
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
    
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process with incident-specific evaluation support."""
        # Run normal test first
        results = super().test(train_epoch, save_metrics, save_results)
        
        # Run incident-specific evaluation if metadata is available
        incident_metrics = self._evaluate_incidents()
        self._save_incident_metrics(incident_metrics)
        
        return results
    
    def _evaluate_incidents(self) -> Optional[Dict]:
        """Evaluate model performance on incident time slots by type."""
        if self.incident_slots is None:
            return None
            
        print("Evaluating on incident time slots by type...")
        
        test_start_idx = self._get_test_data_start_index()
        
        # Get test data range
        dataset = self.test_data_loader.dataset
        total_len = dataset.description['shape'][0]
        test_len = int(total_len * dataset.train_val_test_ratio[2])
        test_end_idx = test_start_idx + test_len
        
        # Collect all incident indices by type
        incident_indices_by_type = {}
        for incident_type, incident_slots in self.incident_slots.items():
            # Filter incident slots to only those in test data range
            test_incident_slots = {slot for slot in incident_slots if test_start_idx <= slot < test_end_idx}
            
            if test_incident_slots:
                # Convert absolute incident slots to relative test indices
                relative_incident_indices = {slot - test_start_idx for slot in test_incident_slots}
                incident_indices_by_type[incident_type] = relative_incident_indices
                print(f"  {incident_type}: {len(relative_incident_indices)} incidents in test range")
        
        if not incident_indices_by_type:
            print("No incident types found in test data range")
            return None
        
        # Run prediction once for all test data
        print("Running predictions for all test data...")
        all_predictions, all_targets, all_inputs, all_indices = [], [], [], []
        
        for data in tqdm(self.test_data_loader, desc="Test data evaluation"):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            
            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()
            
            all_predictions.append(forward_return['prediction'])
            all_targets.append(forward_return['target'])
            all_inputs.append(forward_return['inputs'])
            all_indices.append(data['index'])
        
        # Concatenate all test data
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        print(f"Total test samples: {len(all_predictions)}")
        
        # Evaluate each incident type using pre-computed predictions
        all_incident_metrics = {}
        
        for incident_type, incident_indices in incident_indices_by_type.items():
            print(f"Evaluating {incident_type} incidents...")
            
            # Create mask for this incident type
            incident_mask = torch.tensor([idx.item() in incident_indices for idx in all_indices], dtype=torch.bool)
            
            if not incident_mask.any():
                print(f"  No {incident_type} samples found in test data")
                continue
            
            # Filter predictions for this incident type
            incident_prediction = all_predictions[incident_mask]
            incident_target = all_targets[incident_mask]
            incident_input = all_inputs[incident_mask]
            
            print(f"  Evaluating on {len(incident_prediction)} {incident_type} samples")
            
            # Compute metrics for this incident type
            incident_returns = {
                'prediction': incident_prediction,
                'target': incident_target,
                'inputs': incident_input
            }
            incident_metrics = self.compute_evaluation_metrics(incident_returns)
            
            # Add prefix to distinguish by incident type
            type_metrics = {f'{incident_type}_{k}': v for k, v in incident_metrics.items()}
            all_incident_metrics.update(type_metrics)
        
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
        print("\nIncident Type Performance Summary:")
        incident_types = set()
        for key in incident_metrics.keys():
            if '_horizon_' in key:
                incident_type = key.split('_horizon_')[0]
                incident_types.add(incident_type)
        
        for incident_type in sorted(incident_types):
            print(f"\n{incident_type}:")
            for horizon in [3, 6, 12]:
                mae_key = f'{incident_type}_horizon_{horizon}'
                if mae_key in incident_metrics:
                    mae_value = incident_metrics[mae_key]['MAE']
                    print(f"  Horizon {horizon}: MAE = {mae_value:.4f}")

    def train_iters(self, epoch, iter_index, data):
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # 예측 손실
        pred_loss = self.metric_forward(self.loss, forward_return)
        if self.has_contrastive_loss:
            # Contrastive 손실
            cl_loss = self.contrastive_loss(forward_return)
            total_loss = pred_loss + self.cl_weight * cl_loss
        else:
            total_loss = pred_loss

        self.update_epoch_meter('train/loss', total_loss.item())
        if self.has_contrastive_loss:
            self.update_epoch_meter('train/pred_loss', pred_loss.item())
            self.update_epoch_meter('train/cl_loss', self.cl_weight * cl_loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return total_loss