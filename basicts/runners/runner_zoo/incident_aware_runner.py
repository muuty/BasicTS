import os
import json
from typing import Dict, Optional, Set
import random

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings

#from contrastive.contrastive_loss import MinimalAntiSmoothingLoss


class IncidentAwareRunner(SimpleTimeSeriesForecastingRunner):
    """
    Custom runner that supports incident-specific evaluation.
    """
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.incident_metadata_path = cfg['TEST']['INCIDENT_METADATA_PATH']
        self.incident_slots = self._load_incident_metadata()
        

        # self.has_contrastive_loss = cfg.get('CONTRASTIVE_LOSS', None) is not None
        # if self.has_contrastive_loss:
        #     self.contrastive_loss = cfg['CONTRASTIVE_LOSS']
        #     self.cl_weight = cfg['CONTRASTIVE_LOSS_WEIGHT']
        #     self.register_epoch_meter('train/cl_loss', 'train', '{:.4f}')
        #     self.register_epoch_meter('train/pred_loss', 'train', '{:.4f}')
    
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
        
        if not incident_indices_by_type:
            return None
        
        # Run prediction once for all test data
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
        
        # Evaluate incident performance using pre-computed predictions
        all_incident_metrics = {}
        
        # Helper function to evaluate incident performance
        def evaluate_incident_performance(incident_indices, prefix):
            if not incident_indices:
                return {}

            mask = torch.tensor([idx.item() in incident_indices for idx in all_indices], dtype=torch.bool)
            if not mask.any():
                return {}

            prediction = all_predictions[mask]
            target = all_targets[mask]

            returns = {
                'prediction': prediction,
                'target': target,
                'inputs': all_inputs[mask]
            }

            # Compute metrics using metric_forward
            result_metrics = {}
            for metric_name, metric_func in self.metrics.items():
                metric_val = self.metric_forward(metric_func, returns)
                result_metrics[f'{prefix}_{metric_name}'] = metric_val.item()

            # Also compute per-horizon metrics
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
        
        # Collect all incident indices for overall evaluation
        all_incident_indices = set()
        for incident_indices in incident_indices_by_type.values():
            all_incident_indices.update(incident_indices)
        
        # Evaluate overall incident performance
        if all_incident_indices:
            overall_metrics = evaluate_incident_performance(all_incident_indices, 'overall_incident')
            if overall_metrics:
                all_incident_metrics.update(overall_metrics)
        
        # Evaluate each incident type
        for incident_type, incident_indices in incident_indices_by_type.items():
            type_metrics = evaluate_incident_performance(incident_indices, incident_type)
            if type_metrics:
                all_incident_metrics.update(type_metrics)
        
        # Analyze error distribution and visualize
        self._analyze_incident_errors(
            all_predictions, all_targets, all_inputs, all_indices, 
            all_incident_indices, test_start_idx
        )
        
        return all_incident_metrics
    
    def _analyze_incident_errors(self, all_predictions, all_targets, all_inputs, 
                                  all_indices, all_incident_indices, test_start_idx):
        """Analyze error distribution for incident samples and create visualizations."""
        if not all_incident_indices:
            return
        
        # Create mask for incident samples
        mask = torch.tensor([idx.item() in all_incident_indices for idx in all_indices], dtype=torch.bool)
        if not mask.any():
            return
        
        incident_predictions = all_predictions[mask]
        incident_targets = all_targets[mask]
        incident_inputs = all_inputs[mask]
        incident_indices = all_indices[mask]
        
        # Calculate per-sample MAE (mean over all nodes, features, and time steps)
        # Shape: (num_incident_samples, output_len, num_nodes, num_features)
        per_sample_mae = torch.mean(torch.abs(incident_predictions - incident_targets), dim=(1, 2, 3))
        per_sample_mae_np = per_sample_mae.cpu().numpy()
        
        # Convert to absolute indices for saving
        incident_indices_np = incident_indices.cpu().numpy()
        absolute_indices = incident_indices_np + test_start_idx
        
        # Create output directory
        output_dir = os.path.join(self.ckpt_save_dir, 'incident_error_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot error distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(per_sample_mae_np, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('MAE per Sample', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Error Distribution for Incident Test Samples', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(output_dir, 'error_distribution.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Error distribution plot saved to: {hist_path}")
        
        # 2. Find top 1% error samples
        num_top_samples = max(1, int(len(per_sample_mae_np) * 0.01))
        top_error_indices = np.argsort(per_sample_mae_np)[-num_top_samples:]
        top_error_mae = per_sample_mae_np[top_error_indices]
        top_error_absolute_indices = absolute_indices[top_error_indices]
        
        # Save top error samples info
        top_error_info = {
            'num_total_incident_samples': len(per_sample_mae_np),
            'num_top_1_percent': num_top_samples,
            'top_error_samples': [
                {
                    'absolute_index': int(abs_idx),
                    'relative_index': int(rel_idx),
                    'mae': float(mae)
                }
                for abs_idx, rel_idx, mae in zip(
                    top_error_absolute_indices,
                    incident_indices_np[top_error_indices],
                    top_error_mae
                )
            ]
        }
        
        top_error_path = os.path.join(output_dir, 'top_1_percent_errors.json')
        with open(top_error_path, 'w') as f:
            json.dump(top_error_info, f, indent=4)
        print(f"Top 1% error samples saved to: {top_error_path}")
        
        # 3. Select 40 samples from top errors for visualization
        num_visualize = min(40, num_top_samples)
        if num_visualize > 0:
            selected_indices = random.sample(list(range(num_top_samples)), num_visualize)
            selected_relative_indices = top_error_indices[selected_indices]
            
            # Create visualization grid (8x5 = 40 samples)
            n_rows = 8
            n_cols = 5
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 32))
            fig.suptitle('Top Error Incident Samples: Past, Future, and Prediction', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            dataset = self.test_data_loader.dataset
            input_len = dataset.input_len
            output_len = dataset.output_len
            
            for plot_idx, sample_idx in enumerate(selected_relative_indices):
                row = plot_idx // n_cols
                col = plot_idx % n_cols
                ax = axes[row, col]
                
                # Get data for this sample
                past = incident_inputs[sample_idx].cpu().numpy()  # (input_len, num_nodes, num_features)
                future = incident_targets[sample_idx].cpu().numpy()  # (output_len, num_nodes, num_features)
                pred = incident_predictions[sample_idx].cpu().numpy()  # (output_len, num_nodes, num_features)
                
                # Average over nodes and use first feature (typically flow)
                past_avg = np.mean(past[:, :, 0], axis=1)  # (input_len,)
                future_avg = np.mean(future[:, :, 0], axis=1)  # (output_len,)
                pred_avg = np.mean(pred[:, :, 0], axis=1)  # (output_len,)
                
                # Time axes
                past_time = np.arange(input_len)
                future_time = np.arange(input_len, input_len + output_len)
                
                # Plot
                ax.plot(past_time, past_avg, 'b-', label='Past', linewidth=2)
                ax.plot(future_time, future_avg, 'g-', label='Future (GT)', linewidth=2)
                ax.plot(future_time, pred_avg, 'r--', label='Prediction', linewidth=2, alpha=0.8)
                
                # Vertical line separating past and future
                ax.axvline(x=input_len - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                
                # Title with sample info
                abs_idx = int(absolute_indices[sample_idx])
                mae_val = per_sample_mae_np[sample_idx]
                ax.set_title(f'Sample {abs_idx}\nMAE: {mae_val:.4f}', fontsize=9)
                ax.set_xlabel('Time Step', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Legend only on first subplot
                if plot_idx == 0:
                    ax.legend(loc='upper right', fontsize=7)
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'top_error_samples_visualization.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Top error samples visualization saved to: {viz_path}")
            
            # Also save the selected samples' data
            selected_data = {
                'selected_indices': [int(absolute_indices[idx]) for idx in selected_relative_indices],
                'mae_values': [float(per_sample_mae_np[idx]) for idx in selected_relative_indices]
            }
            selected_data_path = os.path.join(output_dir, 'selected_40_samples.json')
            with open(selected_data_path, 'w') as f:
                json.dump(selected_data, f, indent=4)
            print(f"Selected 40 samples info saved to: {selected_data_path}")
    
    def _save_incident_metrics(self, incident_metrics: Dict):
        """Save incident metrics to JSON file."""
        if incident_metrics is None:
            return
            
        save_path = os.path.join(self.ckpt_save_dir, 'test_incident_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(incident_metrics, f, indent=4)
        
        # Print overall incident performance only
        print("\nOverall Incident Performance:")
        for horizon in [3, 6, 12]:
            mae_key = f'overall_incident_horizon_{horizon}'
            if mae_key in incident_metrics:
                mae_value = incident_metrics[mae_key]['MAE']
                print(f"  Horizon {horizon}: MAE = {mae_value:.4f}")

    def train_iters(self, epoch, iter_index, data):
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # 예측 손실
        pred_loss = self.metric_forward(self.loss, forward_return)
        # if self.has_contrastive_loss:
        #     # Contrastive 손실
        #     cl_loss = self.contrastive_loss(forward_return)
        #     total_loss = pred_loss + self.cl_weight * cl_loss
        # else:
        total_loss = pred_loss

        self.update_epoch_meter('train/loss', total_loss.item())
        # if self.has_contrastive_loss:
        #     self.update_epoch_meter('train/pred_loss', pred_loss.item())
        #     self.update_epoch_meter('train/cl_loss', self.cl_weight * cl_loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return total_loss