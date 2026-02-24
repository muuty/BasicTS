"""
GPT-ST Pre-training Runner.

Custom runner for GPT-ST model that handles:
- Adaptive masking with curriculum learning
- Pre-training with reconstruction loss
- Transition between random and adaptive masking
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from basicts.runners import SimpleTimeSeriesForecastingRunner


class GPTSTPretrainRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for GPT-ST pre-training.

    Extends SimpleTimeSeriesForecastingRunner with:
    - Support for pre-training outputs (reconstruction, mask, probability)
    - Custom loss computation for masked reconstruction
    - Logging of pre-training specific metrics
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Pre-training specific settings
        self.mode = cfg['MODEL'].get('PARAM', {}).get('mode', 'pretrain')
        self.mask_ratio = cfg['MODEL'].get('PARAM', {}).get('mask_ratio', 0.3)
        self.change_epoch = cfg['MODEL'].get('PARAM', {}).get('change_epoch', 10)

        # Loss weights
        self.recon_weight = cfg['TRAIN'].get('LOSS_ARGS', {}).get('recon_weight', 1.0)
        self.class_weight = cfg['TRAIN'].get('LOSS_ARGS', {}).get('class_weight', 0.1)

    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Forward pass with handling for pre-training outputs.

        Args:
            data: Dictionary with 'inputs' and 'target'
            epoch: Current epoch number
            iter_num: Current iteration number
            train: Whether in training mode

        Returns:
            Dictionary with prediction and pre-training outputs
        """
        data = self.preprocessing(data)

        # Prepare data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass
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

        # For pre-training, prediction is reconstruction
        if 'reconstruction' in model_return:
            model_return['prediction'] = model_return['reconstruction']

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            # For pre-training, target is the input (reconstruction task)
            if self.mode == 'pretrain':
                model_return['target'] = self.select_target_features(history_data)
            else:
                model_return['target'] = self.select_target_features(future_data)

        model_return = self.postprocessing(model_return)

        return model_return

    def train_iters(self, epoch: int, iter_index: int,
                    data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration with pre-training specific handling.

        Args:
            epoch: Current epoch
            iter_index: Current iteration index
            data: Data from DataLoader

        Returns:
            Loss tensor
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # Apply curriculum learning if configured
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
            if 'mask' in forward_return:
                forward_return['mask'] = forward_return['mask'][:, :cl_length, :, :]

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
        """Validation iteration.

        Args:
            iter_index: Current iteration index
            data: Data from DataLoader
        """
        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)

        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('val/loss', loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item(), weight)


class GPTSTFinetuneRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for GPT-ST fine-tuning on downstream tasks.

    Uses the EnhanceModel which combines pre-trained encoder with
    a downstream predictor.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Fine-tuning specific settings
        self.freeze_encoder = cfg['MODEL'].get('FREEZE_ENCODER', True)

    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Forward pass for fine-tuning.

        Args:
            data: Dictionary with 'inputs' and 'target'
            epoch: Current epoch number
            iter_num: Current iteration number
            train: Whether in training mode

        Returns:
            Dictionary with prediction
        """
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass
        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
        )

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"Output shape mismatch. Expected [B={batch_size}, L={length}, N={num_nodes}], " \
            f"got {list(model_return['prediction'].shape)[:3]}"

        model_return = self.postprocessing(model_return)

        return model_return
