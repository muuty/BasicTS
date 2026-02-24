"""
STMAE Pre-training Runner.

Handles the pre-training phase of STMAE with:
- Feature masking (temporal patches)
- Structure masking (graph edges)
- Structure and feature reconstruction losses
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from tqdm import tqdm

from basicts.runners.base_tsf_runner import BaseTimeSeriesForecastingRunner


class STMAEPretrainRunner(BaseTimeSeriesForecastingRunner):
    """
    Runner for STMAE pre-training.

    This runner handles the self-supervised pre-training phase where:
    1. Input features are masked using patch_uniform strategy
    2. Graph structure is masked using random-walk based strategy
    3. Model learns to reconstruct both masked features and structure
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Pre-training specific configurations
        self.mask_f_ratio = cfg['MODEL'].get('MASK_F_RATIO', 0.5)
        self.mask_s_ratio = cfg['MODEL'].get('MASK_S_RATIO', 0.3)
        self.patch_length = cfg['MODEL'].get('PATCH_LENGTH', 1)
        self.epoch_wise_mask = cfg['MODEL'].get('EPOCH_WISE_MASK', False)

        # Loss weights
        self.sl_weight = cfg['TRAIN'].get('SL_WEIGHT', 1.0)
        self.fl_weight = cfg['TRAIN'].get('FL_WEIGHT', 1.0)

        # Feature selection
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)

        # Current epoch masks (for epoch-wise masking)
        self._current_f_mask = None
        self._current_s_mask = None

    def init_training(self, cfg: Dict):
        """Initialize training components."""
        super().init_training(cfg)

        # Register additional meters for pre-training
        self.register_epoch_meter('train/s_loss', 'train', '{:.4f}')
        self.register_epoch_meter('train/f_loss', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        """Initialize validation components."""
        super().init_validation(cfg)

        # Register additional meters
        self.register_epoch_meter('val/s_loss', 'val', '{:.4f}')
        self.register_epoch_meter('val/f_loss', 'val', '{:.4f}')

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess input data."""
        if self.scaler is not None:
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
            if 'target' in input_data:
                input_data['target'] = self.scaler.transform(input_data['target'])
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess output data."""
        if self.scaler is not None and self.scaler.rescale:
            if 'reconstruction' in input_data:
                input_data['reconstruction'] = self.scaler.inverse_transform(
                    input_data['reconstruction']
                )
            if 'target' in input_data:
                input_data['target'] = self.scaler.inverse_transform(input_data['target'])
        return input_data

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features based on configuration."""
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features based on configuration."""
        if self.target_features is not None:
            data = data[:, :, :, self.target_features]
        return data

    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        super().on_epoch_start(epoch) if hasattr(super(), 'on_epoch_start') else None

        # Generate epoch-wise masks if enabled
        if self.epoch_wise_mask and hasattr(self.model, 'feature_masking'):
            self._generate_epoch_masks()

    def _generate_epoch_masks(self):
        """Generate masks once per epoch for consistency."""
        # Get a sample batch shape
        B = 32  # Placeholder batch size
        T = self.model.input_len if hasattr(self.model, 'input_len') else 12
        N = self.model.num_nodes if hasattr(self.model, 'num_nodes') else 100
        D = self.model.input_dim if hasattr(self.model, 'input_dim') else 1

        device = next(self.model.parameters()).device
        x_holder = torch.rand(B, T, N, D, device=device)

        # Generate feature mask
        if hasattr(self.model, 'feature_masking'):
            _, self._current_f_mask = self.model.feature_masking(
                x_holder[..., :1], self.mask_f_ratio
            )

        # Generate structure mask
        if hasattr(self.model, 'structure_masking') and hasattr(self.model, 'get_support'):
            support = self.model.get_support()
            self._current_s_mask = self.model.structure_masking(
                support, self.mask_s_ratio
            )

    def forward(
        self,
        data: Dict,
        epoch: Optional[int] = None,
        iter_num: Optional[int] = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """
        Forward pass for pre-training.

        Args:
            data: Input data dictionary with 'inputs' key
            epoch: Current epoch
            iter_num: Current iteration
            train: Training mode flag

        Returns:
            Dictionary with loss, predictions, and masks
        """
        data = self.preprocessing(data)

        # Get history data
        history_data = data['inputs']
        history_data = self.to_running_device(history_data)

        # Select features
        history_data = self.select_input_features(history_data)

        batch_size, seq_len, num_nodes, _ = history_data.shape

        # Prepare kwargs for model
        model_kwargs = {
            'mask_s': self.mask_s_ratio if train else 0,
            'mask_f': self.mask_f_ratio if train else 0,
            'sl_weight': self.sl_weight,
            'fl_weight': self.fl_weight,
            'batch_seen': iter_num,
            'epoch': epoch,
            'train': train,
        }

        # Use epoch-wise masks if available
        if self.epoch_wise_mask and self._current_f_mask is not None:
            # Adjust mask batch size if needed
            if self._current_f_mask.shape[0] != batch_size:
                model_kwargs['f_mask'] = self._current_f_mask[:batch_size]
            else:
                model_kwargs['f_mask'] = self._current_f_mask

        if self.epoch_wise_mask and self._current_s_mask is not None:
            model_kwargs['s_mask'] = self._current_s_mask

        # Forward through model
        model_return = self.model(
            history_data=history_data,
            future_data=None,
            **model_kwargs
        )

        # Ensure required keys are present
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(history_data)
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'prediction' not in model_return:
            # For pre-training, prediction is the reconstruction
            model_return['prediction'] = model_return.get('reconstruction', history_data)

        model_return = self.postprocessing(model_return)

        return model_return

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration."""
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # Get loss (already computed in model)
        if 'loss' in forward_return:
            loss = forward_return['loss']
        else:
            loss = self.metric_forward(self.loss, forward_return)

        # Update meters
        batch_size = forward_return['target'].shape[0]
        self.update_epoch_meter('train/loss', loss.item(), batch_size)

        # Update component losses if available
        if 's_loss' in forward_return:
            self.update_epoch_meter('train/s_loss', forward_return['s_loss'].item()
                                   if torch.is_tensor(forward_return['s_loss'])
                                   else forward_return['s_loss'], batch_size)
        if 'f_loss' in forward_return:
            self.update_epoch_meter('train/f_loss', forward_return['f_loss'].item()
                                   if torch.is_tensor(forward_return['f_loss'])
                                   else forward_return['f_loss'], batch_size)

        # Compute metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), batch_size)

        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration."""
        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)

        # Get loss
        if 'loss' in forward_return:
            loss = forward_return['loss']
        else:
            loss = self.metric_forward(self.loss, forward_return)

        batch_size = forward_return['target'].shape[0]
        self.update_epoch_meter('val/loss', loss.item(), batch_size)

        # Update component losses
        if 's_loss' in forward_return:
            self.update_epoch_meter('val/s_loss', forward_return['s_loss'].item()
                                   if torch.is_tensor(forward_return['s_loss'])
                                   else forward_return['s_loss'], batch_size)
        if 'f_loss' in forward_return:
            self.update_epoch_meter('val/f_loss', forward_return['f_loss'].item()
                                   if torch.is_tensor(forward_return['f_loss'])
                                   else forward_return['f_loss'], batch_size)

        # Compute metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item(), batch_size)

    def save_pretrained_encoder(self, save_path: str = None):
        """
        Save pre-trained encoder weights for fine-tuning.

        Args:
            save_path: Path to save encoder weights. If None, saves to checkpoint dir.
        """
        if save_path is None:
            save_path = os.path.join(self.ckpt_save_dir, 'pretrained_encoder.pt')

        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'node_embeddings': self.model.node_embeddings.data if hasattr(self.model, 'node_embeddings') else None,
            'mask_token': self.model.mask_token.data if hasattr(self.model, 'mask_token') else None,
        }, save_path)

        self.logger.info(f'Saved pre-trained encoder to {save_path}')

    def on_training_end(self):
        """Called at the end of training."""
        super().on_training_end() if hasattr(super(), 'on_training_end') else None

        # Save pre-trained encoder
        self.save_pretrained_encoder()
