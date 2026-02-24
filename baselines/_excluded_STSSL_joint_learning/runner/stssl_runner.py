"""
ST-SSL Runner for BasiCTS.

Handles:
- Loading adjacency matrix for graph operations
- Two-view training with augmentations
- Combined loss computation (prediction + temporal + spatial)
"""
from typing import Dict, Optional, Union, Tuple
import os

import torch
import numpy as np

from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import load_adj
from basicts.metrics import masked_mae


class STSSLRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for ST-SSL model.

    This runner handles:
    1. Loading and passing adjacency matrix to the model
    2. Computing combined losses (prediction + SSL losses)
    3. Supporting different training modes (pretrain, finetune, end2end)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Load adjacency matrix
        self.adj_mx = None
        self._load_adjacency_matrix(cfg)

        # Training mode
        self.ssl_mode = cfg.get('MODEL', {}).get('PARAM', {}).get('mode', 'end2end')

        # Loss weights
        self.loss_weights = cfg.get('MODEL', {}).get('PARAM', {}).get('loss_weights', [1.0, 1.0, 1.0])

        # Set adjacency matrix to model after it's built
        self._adj_mx_set = False

    def _load_adjacency_matrix(self, cfg: Dict):
        """Load adjacency matrix from config."""
        # Try different config locations for adj path
        adj_path = None

        # Check GRAPH config
        graph_cfg = cfg.get('GRAPH', {})
        if 'adj_path' in graph_cfg:
            adj_path = graph_cfg['adj_path']

        # Check MODEL.PARAM
        if adj_path is None:
            model_param = cfg.get('MODEL', {}).get('PARAM', {})
            if 'adj_path' in model_param:
                adj_path = model_param['adj_path']

        # Check DATASET.PARAM
        if adj_path is None:
            dataset_param = cfg.get('DATASET', {}).get('PARAM', {})
            dataset_name = dataset_param.get('dataset_name', '')
            if dataset_name:
                # Try to construct default adj path
                potential_path = f'datasets/{dataset_name}/adj_mx.pkl'
                if os.path.exists(potential_path):
                    adj_path = potential_path

        if adj_path and os.path.exists(adj_path):
            try:
                adj_mx, _ = load_adj(adj_path, 'doubletransition')
                if isinstance(adj_mx, list):
                    adj_mx = adj_mx[0]
                if isinstance(adj_mx, np.ndarray):
                    adj_mx = torch.from_numpy(adj_mx).float()
                self.adj_mx = adj_mx
                self.logger.info(f"Loaded adjacency matrix from {adj_path}, shape: {adj_mx.shape}")
            except Exception as e:
                self.logger.warning(f"Failed to load adjacency matrix from {adj_path}: {e}")
        else:
            self.logger.warning(f"No adjacency matrix found. Using identity matrix.")

    def _ensure_adj_mx_set(self):
        """Ensure adjacency matrix is set on model."""
        if not self._adj_mx_set and self.adj_mx is not None:
            if hasattr(self.model, 'set_adj_matrix'):
                self.model.set_adj_matrix(self.adj_mx)
                self._adj_mx_set = True
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'set_adj_matrix'):
                self.model.model.set_adj_matrix(self.adj_mx)
                self._adj_mx_set = True

    def forward(
            self,
            data: Dict,
            epoch: Optional[int] = None,
            iter_num: Optional[int] = None,
            train: bool = True,
            **kwargs
    ) -> Dict:
        """
        Forward pass for ST-SSL.

        Args:
            data: Dictionary containing 'target' and 'inputs'
            epoch: Current epoch number
            iter_num: Current iteration number
            train: Whether in training mode

        Returns:
            Dictionary containing predictions and optional SSL outputs
        """
        # Ensure adj_mx is set
        self._ensure_adj_mx_set()

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Prepare adjacency matrix
        adj_mx = None
        if self.adj_mx is not None:
            adj_mx = self.adj_mx.to(history_data.device)

        # Forward pass through the model
        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            adj_mx=adj_mx
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"Output shape mismatch. Expected [{batch_size}, {length}, {num_nodes}, C], got {model_return['prediction'].shape}"

        model_return = self.postprocessing(model_return)

        return model_return

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """
        Training iteration with ST-SSL specific loss computation.

        Args:
            epoch: Current epoch
            iter_index: Current iteration index
            data: Data from DataLoader

        Returns:
            loss: Combined loss value
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # Apply curriculum learning if enabled
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        # Compute prediction loss
        pred_loss = self.metric_forward(self.loss, forward_return)

        # Compute SSL losses if available
        total_loss = pred_loss
        ssl_losses = {}

        if 'repr1' in forward_return and 'repr2' in forward_return:
            repr1 = forward_return['repr1']
            repr2 = forward_return['repr2']

            # Get the actual model (handle DataParallel wrapping)
            model = self.model
            if hasattr(model, 'module'):
                model = model.module
            if hasattr(model, 'model'):
                model = model.model

            # Compute SSL losses
            if hasattr(model, 'compute_ssl_losses'):
                temporal_loss, spatial_loss = model.compute_ssl_losses(repr1, repr2)
                ssl_losses['temporal'] = temporal_loss
                ssl_losses['spatial'] = spatial_loss

                # Combine losses based on mode
                if self.ssl_mode == 'pretrain':
                    total_loss = (
                            self.loss_weights[1] * temporal_loss +
                            self.loss_weights[2] * spatial_loss
                    )
                elif self.ssl_mode == 'end2end':
                    total_loss = (
                            self.loss_weights[0] * pred_loss +
                            self.loss_weights[1] * temporal_loss +
                            self.loss_weights[2] * spatial_loss
                    )
                # else: finetune mode uses only pred_loss

        # Update meters
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', total_loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)

        return total_loss

    def init_training(self, cfg: Dict):
        """Initialize training with SSL-specific meters."""
        super().init_training(cfg)

        # Register SSL loss meters if in SSL mode
        if self.ssl_mode in ['pretrain', 'end2end']:
            self.register_epoch_meter('train/temporal_loss', 'train', '{:.4f}')
            self.register_epoch_meter('train/spatial_loss', 'train', '{:.4f}')
