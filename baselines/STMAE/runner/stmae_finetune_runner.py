"""
STMAE Fine-tuning Runner.

Handles the fine-tuning phase of STMAE for forecasting:
- Loads pre-trained encoder
- Attaches forecasting decoder
- Supports frozen or fine-tuned encoder
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from basicts.runners.base_tsf_runner import BaseTimeSeriesForecastingRunner


class STMAEFinetuneRunner(BaseTimeSeriesForecastingRunner):
    """
    Runner for STMAE fine-tuning.

    This runner handles the supervised fine-tuning phase where:
    1. Pre-trained encoder is loaded (optionally frozen)
    2. Forecasting decoder is trained to produce predictions
    3. Standard forecasting loss (MAE) is used
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Fine-tuning specific configurations
        self.pretrained_path = cfg['MODEL'].get('PRETRAINED_PATH', None)
        self.freeze_encoder = cfg['MODEL'].get('FREEZE_ENCODER', False)

        # Learning rate configurations (can differ for encoder vs decoder)
        self.lr_encoder = cfg['TRAIN'].get('LR_ENCODER', None)
        self.lr_decoder = cfg['TRAIN'].get('LR_DECODER', None)

        # Feature selection
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)

        # Load pre-trained weights if provided
        if self.pretrained_path is not None:
            self._load_pretrained_encoder()

    def _load_pretrained_encoder(self):
        """Load pre-trained encoder weights."""
        if not os.path.exists(self.pretrained_path):
            self.logger.warning(f'Pre-trained encoder not found at {self.pretrained_path}')
            return

        checkpoint = torch.load(self.pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Load encoder weights (handle STMAEForecaster wrapper)
        if hasattr(self.model, 'encoder'):
            # Model is STMAEForecaster with encoder attribute
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k[8:]] = v  # Remove 'encoder.' prefix
                else:
                    encoder_state_dict[k] = v

            missing_keys, unexpected_keys = self.model.encoder.load_state_dict(
                encoder_state_dict, strict=False
            )
        else:
            # Model is STMAE directly
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )

        self.logger.info(f'Loaded pre-trained encoder from {self.pretrained_path}')
        if missing_keys:
            self.logger.warning(f'Missing keys: {missing_keys}')
        if unexpected_keys:
            self.logger.warning(f'Unexpected keys: {unexpected_keys}')

        # Freeze encoder if specified
        if self.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self.logger.info('Encoder parameters frozen')
        else:
            # Freeze all except decoder
            for name, param in self.model.named_parameters():
                if 'decoder' not in name.lower():
                    param.requires_grad = False
            self.logger.info('Non-decoder parameters frozen')

    def build_optimizer(self, cfg: Dict):
        """Build optimizer with optional separate learning rates."""
        # If separate learning rates are specified
        if self.lr_encoder is not None and self.lr_decoder is not None:
            encoder_params = []
            decoder_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'decoder' in name.lower():
                    decoder_params.append(param)
                else:
                    encoder_params.append(param)

            param_groups = []
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': self.lr_encoder,
                })
            if decoder_params:
                param_groups.append({
                    'params': decoder_params,
                    'lr': self.lr_decoder,
                })

            optimizer_type = cfg['TRAIN']['OPTIM']['TYPE']
            optimizer_params = cfg['TRAIN']['OPTIM'].get('PARAM', {})

            # Remove lr from params since we set it per group
            optimizer_params = {k: v for k, v in optimizer_params.items() if k != 'lr'}

            return optimizer_type(param_groups, **optimizer_params)

        # Default optimizer building
        return super().build_optimizer(cfg) if hasattr(super(), 'build_optimizer') else None

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess input data."""
        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess output data."""
        # Rescale predictions and targets
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        # Subset forecasting (specific time series)
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

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

    def forward(
        self,
        data: Dict,
        epoch: Optional[int] = None,
        iter_num: Optional[int] = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        """
        Forward pass for fine-tuning.

        Args:
            data: Input data dictionary with 'inputs' and 'target' keys
            epoch: Current epoch
            iter_num: Current iteration
            train: Training mode flag

        Returns:
            Dictionary with prediction, target, and inputs
        """
        data = self.preprocessing(data)

        # Get data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # For non-training, don't use future target values
            future_data_4_dec = torch.empty_like(future_data_4_dec)

        # Forward through model
        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"Output shape mismatch. Expected [{batch_size}, {length}, {num_nodes}, ...], " \
            f"got {list(model_return['prediction'].shape)}"

        model_return = self.postprocessing(model_return)

        return model_return

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration."""
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # Apply curriculum learning if configured
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        # Compute loss
        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', loss.item(), weight)

        # Compute metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)

        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration."""
        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)

        # Compute loss
        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('val/loss', loss.item(), weight)

        # Compute metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item(), weight)

    def save_finetuned_model(self, save_path: str = None):
        """
        Save fine-tuned model weights.

        Args:
            save_path: Path to save model weights. If None, saves to checkpoint dir.
        """
        if save_path is None:
            save_path = os.path.join(self.ckpt_save_dir, 'finetuned_model.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, save_path)

        self.logger.info(f'Saved fine-tuned model to {save_path}')
