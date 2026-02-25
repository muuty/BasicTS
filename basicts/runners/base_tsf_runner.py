import functools
import inspect
import json
import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict
from easytorch.utils import master_only
from tqdm import tqdm

from basicts.data.simple_inference_dataset import TimeSeriesInferenceDataset

from ..metrics import (masked_mae, masked_mape, masked_mse, masked_rmse,
                       masked_wape)
from .base_epoch_runner import BaseEpochRunner


class BaseTimeSeriesForecastingRunner(BaseEpochRunner):
    """
    Runner for multivariate time series forecasting tasks.

    Features:
        - Supports evaluation at pre-defined horizons (optional) and overall performance assessment.
        - Metrics: MAE, RMSE, MAPE, WAPE, and MSE. Customizable. The best model is selected based on the smallest MAE on the validation set.
        - Supports `setup_graph` for models that operate similarly to TensorFlow.
        - Default loss function is MAE (masked_mae), but it can be customized.
        - Supports curriculum learning.
        - Users only need to implement the `forward` function.

    Customization:
        - Model:
            - Args:
                - history_data (torch.Tensor): Historical data with shape [B, L, N, C], 
                  where B is the batch size, L is the sequence length, N is the number of nodes, 
                  and C is the number of features.
                - future_data (torch.Tensor or None): Future data with shape [B, L, N, C]. 
                  Can be None if there is no future data available.
                - batch_seen (int): The number of batches seen so far.
                - epoch (int): The current epoch number.
                - train (bool): Indicates whether the model is in training mode.
            - Return:
                - Dict or torch.Tensor:
                    - If returning a Dict, it must contain the 'prediction' key. Other keys are optional and will be passed to the loss and metric functions.
                    - If returning a torch.Tensor, it should represent the model's predictions, with shape [B, L, N, C].

        - Loss & Metrics (optional):
            - Args:
                - prediction (torch.Tensor): Model's predictions, with shape [B, L, N, C].
                - target (torch.Tensor): Ground truth data, with shape [B, L, N, C].
                - null_val (float): The value representing missing data in the dataset.
                - Other args (optional): Additional arguments will be matched with keys in the model's return dictionary, if applicable.
            - Return:
                - torch.Tensor: The computed loss or metric value.

        - Dataset (optional):
            - Return: The returned data will be passed to the `forward` function as the `data` argument.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # setup graph flag
        self.need_setup_graph = cfg['MODEL'].get('SETUP_GRAPH', False)

        # initialize scaler
        self.scaler = self.build_scaler(cfg)

        # define loss function
        self.loss = cfg['TRAIN']['LOSS']
        self.loss_args = cfg['TRAIN'].get('LOSS_ARGS', {})

        # define metrics
        self.metrics = cfg.get('METRICS', {}).get('FUNCS', {
                                                            'MAE': masked_mae, 
                                                            'RMSE': masked_rmse,
                                                            'MAPE': masked_mape, 
                                                            'WAPE': masked_wape, 
                                                            'MSE': masked_mse
                                                        })
        self.target_metrics = cfg.get('METRICS', {}).get('TARGET', 'loss')
        self.metrics_best = cfg.get('METRICS', {}).get('BEST', 'min')
        assert self.target_metrics in self.metrics or self.target_metrics == 'loss', f'Target metric {self.target_metrics} not found in metrics.'
        assert self.metrics_best in ['min', 'max'], f'Invalid best metric {self.metrics_best}.'
        # handle null values in datasets, e.g., 0.0 or np.nan.
        self.null_val = cfg.get('METRICS', {}).get('NULL_VAL', np.nan)

        # curriculum learning setup
        self.cl_param = cfg['TRAIN'].get('CL', None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg['TRAIN'].CL.get('WARM_EPOCHS', 0)
            self.cl_epochs = cfg['TRAIN'].CL.get('CL_EPOCHS')
            self.prediction_length = cfg['TRAIN'].CL.get('PREDICTION_LENGTH')
            self.cl_step_size = cfg['TRAIN'].CL.get('STEP_SIZE', 1)

        # Eealuation settings
        self.if_evaluate_on_gpu = cfg.get('EVAL', EasyDict()).get('USE_GPU', True)
        self.evaluation_horizons = [_ - 1 for _ in cfg.get('EVAL', EasyDict()).get('HORIZONS', [])]
        assert len(self.evaluation_horizons) == 0 or min(self.evaluation_horizons) >= 0, 'The horizon should start counting from 1.'

        # Noise robustness evaluation
        noise_cfg = cfg.get('EVAL', EasyDict()).get('NOISE_ROBUSTNESS', False)
        self._noise_eval_enabled = bool(noise_cfg)
        if self._noise_eval_enabled:
            if isinstance(noise_cfg, dict):
                self._noise_eval_configs = noise_cfg.get('configs', None)
                self._noise_eval_channels = noise_cfg.get('physical_channels', [0, 1, 2])
            else:
                self._noise_eval_configs = None  # use defaults
                self._noise_eval_channels = [0, 1, 2]
            self._noise_eval_dataset = cfg['DATASET']['NAME']

        # For saving test results
        self._inputs_memmap = None
        self._prediction_memmap = None
        self._target_memmap = None

    def build_scaler(self, cfg: Dict):
        """Build scaler.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Scaler instance or None if no scaler is declared.
        """

        if 'SCALER' in cfg:
            return cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
        return None

    def setup_graph(self, cfg: Dict, train: bool):
        """Setup all parameters and the computation graph.

        Some models (e.g., DCRNN, GTS) require creating parameters during the first forward pass, similar to TensorFlow.

        Args:
            cfg (Dict): Configuration.
            train (bool): Whether the setup is for training or inference.
        """

        dataloader = self.build_test_data_loader(cfg=cfg) if not train else self.build_train_data_loader(cfg=cfg)
        data = next(iter(dataloader))  # get the first batch
        if not train:
            self.model.eval()
        self.forward(data=data, epoch=1, iter_num=0, train=train)

    def count_parameters(self):
        """Count the number of parameters in the model."""

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of parameters: {num_parameters}')

    def init_training(self, cfg: Dict):
        """Initialize training components, including loss, meters, etc.

        Args:
            cfg (Dict): Configuration.
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=True)
            self.need_setup_graph = False

        super().init_training(cfg)
        self.count_parameters()

        self.register_epoch_meter('train/loss', 'train', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'train/{key}', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        """Initialize validation components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        super().init_validation(cfg)
        self.register_epoch_meter('val/loss', 'val', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'val/{key}', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):
        """Initialize test components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=False)
            self.need_setup_graph = False

        super().init_test(cfg)
        self.register_epoch_meter('test/loss', 'test', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'test/{key}', 'test', '{:.4f}')
        # Register metrics for each evaluation horizons
        for i in self.evaluation_horizons:
            for key in self.metrics:
                self.register_epoch_meter(f'test/{key}@h{i+1}', f'test @ horizon {i+1}', '{:.4f}')

    def build_train_dataset(self, cfg: Dict):
        """Build the training dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed training dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['mode'] = 'train'
            dataset = cfg['TRAIN']['DATA']['DATASET']['TYPE'](**cfg['TRAIN']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            self.iter_per_epoch = math.ceil(len(dataset) / batch_size)
        else:
            dataset = cfg['DATASET']['TYPE'](mode='train', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    def build_val_dataset(self, cfg: Dict):
        """Build the validation dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed validation dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['mode'] = 'valid'
            dataset = cfg['VAL']['DATA']['DATASET']['TYPE'](**cfg['VAL']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='valid', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')

        return dataset

    def build_test_dataset(self, cfg: Dict):
        """Build the test dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed test dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['mode'] = 'test'
            dataset = cfg['TEST']['DATA']['DATASET']['TYPE'](**cfg['TEST']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='test', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')

        return dataset

    def build_inference_dataset(self, cfg: Dict, input_data: Union[str, list]):
        """Build the inference dataset.

        Args:
            cfg (Dict): Configuration.
            input_data (Union[str, list]): The input data file path or data list for inference.
        Returns:
            Dataset: The constructed inference dataset.
        """

        dataset = TimeSeriesInferenceDataset(dataset=input_data, logger=self.logger, **cfg['DATASET']['PARAM'])
        self.logger.info(f'Inference dataset length: {len(dataset)}')

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level for curriculum learning.

        Args:
            epoch (int, optional): Current epoch if in training process; None otherwise. Defaults to None.

        Returns:
            int: Task level for the current epoch.
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still in warm-up phase
            cl_length = self.prediction_length
        else:
            progress = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(progress, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 
        Note: The outputs are not re-scaled.

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        raise NotImplementedError()

    def metric_forward(self, metric_func, args: Dict) -> torch.Tensor:
        """Compute metrics using the given metric function.

        Args:
            metric_func (function or functools.partial): Metric function.
            args (Dict): Arguments for metrics computation.

        Returns:
            torch.Tensor: Computed metric value.
        """

        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}

        # add loss args
        if self.loss_args:
            for k, v in self.loss_args.items():
                if k in covariate_names and k not in args:
                    args[k] = v

        if isinstance(metric_func, functools.partial):
            if 'null_val' not in metric_func.keywords and 'null_val' in covariate_names: # null_val is required but not provided
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        elif callable(metric_func):
            if 'null_val' in covariate_names: # null_val is required
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_func)}')
        return metric_item

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('val/loss', loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item(), weight)

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.

        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        # Collect per-sample and per-node MAE for robustness metrics
        all_sample_mae = []
        all_node_mae = []  # Will be aggregated per node
        all_masked_ae_sum = []  # For masked per-node MAE (skip zero targets)
        all_nonzero_count = []  # Count of non-zero targets per node

        for batch_idx, data in enumerate(tqdm(self.test_data_loader)):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            weight = self._get_metric_weight(forward_return['target'])
            self.update_epoch_meter('test/loss', loss.item(), weight)

            if not self.if_evaluate_on_gpu:
                pred = forward_return['prediction'].detach().cpu()
                target = forward_return['target'].detach().cpu()
            else:
                pred = forward_return['prediction']
                target = forward_return['target']
            if save_results:
                batch_data = {
                    'prediction': forward_return['prediction'].detach().cpu().numpy(),
                    'target': forward_return['target'].detach().cpu().numpy(),
                    'inputs': forward_return['inputs'].detach().cpu().numpy()
                }
                self._save_test_results(batch_idx, batch_data)

            # Compute per-sample MAE for robustness metrics
            # Shape: (batch_size,) - mean over time, nodes, and features
            batch_sample_mae = torch.mean(torch.abs(pred - target), dim=(1, 2, 3))
            all_sample_mae.append(batch_sample_mae.cpu())

            # Compute per-node MAE for robustness metrics
            # Shape: (batch_size, num_nodes) - mean over time and features
            batch_node_mae = torch.mean(torch.abs(pred - target), dim=(1, 3))
            all_node_mae.append(batch_node_mae.cpu())

            # Accumulate masked per-node MAE (skip zero targets)
            batch_ae = torch.abs(pred - target)  # (B, T, N, C)
            batch_nonzero = (target != 0)  # (B, T, N, C)
            all_masked_ae_sum.append((batch_ae * batch_nonzero).sum(dim=(1, 3)).cpu())  # (B, N)
            all_nonzero_count.append(batch_nonzero.sum(dim=(1, 3)).cpu())  # (B, N)

            # evaluation on specific timesteps
            for i in self.evaluation_horizons:
                pred_h = pred[:, i, :, :]
                target_h = target[:, i, :, :]
                weight_h = self._get_metric_weight(target_h)

                for metric_name, metric_func in self.metrics.items():
                    if metric_name.lower() == 'mase':
                        continue  # MASE needs to be calculated after all horizons
                    metric_val = self.metric_forward(metric_func, {'prediction': pred_h, 'target': target_h})
                    self.update_epoch_meter(f'test/{metric_name}@h{i+1}', metric_val.item(), weight_h)

            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': target})
                self.update_epoch_meter(f'test/{metric_name}', metric_item.item(), weight)

        # Compute robustness metrics
        all_sample_mae = torch.cat(all_sample_mae, dim=0).numpy()
        all_node_mae = torch.cat(all_node_mae, dim=0).numpy()  # Shape: (num_samples, num_nodes)
        masked_ae_sum = torch.cat(all_masked_ae_sum, dim=0).numpy().sum(axis=0)  # (num_nodes,)
        nonzero_count = torch.cat(all_nonzero_count, dim=0).numpy().sum(axis=0)  # (num_nodes,)
        self._robustness_metrics = self._compute_robustness_metrics(
            all_sample_mae, all_node_mae, masked_ae_sum, nonzero_count
        )

        # Noise robustness evaluation (test-time noise injection)
        if self._noise_eval_enabled:
            clean_node_mae = np.mean(all_node_mae, axis=0)  # (num_nodes,)
            self._noise_robustness = self._eval_noise_robustness(clean_node_mae)

        if save_metrics:
            self._save_test_metrics()

    def _compute_robustness_metrics(self, sample_mae: np.ndarray, node_mae: np.ndarray,
                                     masked_ae_sum: np.ndarray = None,
                                     nonzero_count: np.ndarray = None) -> Dict:
        """Compute robustness metrics (worst K% MAE) for both samples and nodes.

        Args:
            sample_mae (np.ndarray): Per-sample MAE values with shape (num_samples,).
            node_mae (np.ndarray): Per-node MAE values with shape (num_samples, num_nodes).
            masked_ae_sum (np.ndarray): Sum of |pred-target| where target!=0, per node. Shape: (num_nodes,).
            nonzero_count (np.ndarray): Count of non-zero targets per node. Shape: (num_nodes,).

        Returns:
            Dict: Robustness metrics including worst 1%, 5%, 10% MAE for samples and nodes.
        """
        robustness = {}

        # ===== Per-Sample Robustness =====
        num_samples = len(sample_mae)
        sorted_sample_mae = np.sort(sample_mae)[::-1]  # Sort descending (worst first)

        sample_metrics = {}
        # Worst K% metrics
        for pct in [1, 5, 10]:
            k = max(1, int(num_samples * pct / 100))
            worst_k_mae = np.mean(sorted_sample_mae[:k])
            sample_metrics[f'worst_{pct}pct_MAE'] = float(worst_k_mae)

        # Additional statistics
        sample_metrics['max_MAE'] = float(sorted_sample_mae[0])
        sample_metrics['median_MAE'] = float(np.median(sample_mae))
        sample_metrics['std_MAE'] = float(np.std(sample_mae))
        sample_metrics['num_samples'] = num_samples
        robustness['per_sample'] = sample_metrics

        # ===== Per-Node Robustness =====
        # Compute mean MAE per node across all samples
        # node_mae shape: (num_samples, num_nodes)
        mean_node_mae = np.mean(node_mae, axis=0)  # Shape: (num_nodes,)
        num_nodes = len(mean_node_mae)
        sorted_node_mae = np.sort(mean_node_mae)[::-1]  # Sort descending (worst first)

        node_metrics = {}
        # Worst K% node metrics
        for pct in [1, 5, 10]:
            k = max(1, int(num_nodes * pct / 100))
            worst_k_mae = np.mean(sorted_node_mae[:k])
            node_metrics[f'worst_{pct}pct_MAE'] = float(worst_k_mae)

        # Additional statistics
        node_metrics['max_MAE'] = float(sorted_node_mae[0])
        node_metrics['min_MAE'] = float(sorted_node_mae[-1])
        node_metrics['median_MAE'] = float(np.median(mean_node_mae))
        node_metrics['std_MAE'] = float(np.std(mean_node_mae))  # Node-wise error variance
        node_metrics['num_nodes'] = num_nodes

        # Worst and best node indices
        node_metrics['worst_node_idx'] = int(np.argmax(mean_node_mae))
        node_metrics['best_node_idx'] = int(np.argmin(mean_node_mae))

        # Worst 1% node indices (sorted by MAE descending)
        sorted_node_indices = np.argsort(mean_node_mae)[::-1]
        k1 = max(1, int(num_nodes * 1 / 100))
        node_metrics['worst_1pct_node_indices'] = [int(i) for i in sorted_node_indices[:k1]]
        node_metrics['worst_1pct_node_maes'] = [float(mean_node_mae[i]) for i in sorted_node_indices[:k1]]

        robustness['per_node'] = node_metrics

        # ===== Per-Category Robustness (by target zero rate) =====
        if masked_ae_sum is not None and nonzero_count is not None:
            safe_count = np.maximum(nonzero_count, 1)
            masked_node_mae = masked_ae_sum / safe_count  # (num_nodes,)

            # Compute per-node zero rate from target counts
            # Use nonzero_count relative to max across nodes as proxy for zero rate
            max_nonzero = nonzero_count.max()
            zero_rate = 1.0 - nonzero_count / max_nonzero if max_nonzero > 0 else np.zeros(num_nodes)

            categories = {
                'dead_gt90pct': zero_rate > 0.9,
                'major_fail_50_90pct': (zero_rate > 0.5) & (zero_rate <= 0.9),
                'partial_fail_5_50pct': (zero_rate > 0.05) & (zero_rate <= 0.5),
                'functional_lt5pct': zero_rate <= 0.05,
            }

            cat_metrics = {}
            for cat_name, mask in categories.items():
                n = int(mask.sum())
                if n == 0:
                    cat_metrics[cat_name] = {'num_nodes': 0, 'raw_mae': None, 'masked_mae': None}
                else:
                    cat_metrics[cat_name] = {
                        'num_nodes': n,
                        'raw_mae': float(mean_node_mae[mask].mean()),
                        'masked_mae': float(masked_node_mae[mask].mean()),
                    }
            cat_metrics['overall_masked_mae'] = float(
                masked_ae_sum.sum() / max(nonzero_count.sum(), 1)
            )
            robustness['per_category'] = cat_metrics

        return robustness

    def _save_test_metrics(self):
        """Save test metrics to JSON file. Override this to add custom metrics."""
        metrics_results = {}
        metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}

        # Add robustness metrics
        if hasattr(self, '_robustness_metrics') and self._robustness_metrics:
            metrics_results['robustness'] = self._robustness_metrics

        # Add noise robustness metrics
        if hasattr(self, '_noise_robustness') and self._noise_robustness:
            metrics_results['noise_robustness'] = self._noise_robustness

        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)

    @torch.no_grad()
    def _eval_noise_robustness(self, clean_node_mae: np.ndarray) -> Dict:
        """Evaluate model robustness under test-time noise injection.

        For each noise config, injects noise into test inputs and measures
        MAE degradation on functional, corrupted, and healthy (uncorrupted) nodes.

        Args:
            clean_node_mae: Per-node MAE from clean test, shape (num_nodes,).

        Returns:
            Dict with clean baseline, and per-config degradation metrics.
        """
        from .noise_eval import (DEFAULT_NOISE_CONFIGS, inject_noise,
                                 select_corrupt_nodes, load_functional_indices, SEED)

        configs = self._noise_eval_configs or DEFAULT_NOISE_CONFIGS
        physical_channels = self._noise_eval_channels
        num_nodes = len(clean_node_mae)

        functional = load_functional_indices(self._noise_eval_dataset, num_nodes)
        clean_func_mae = float(clean_node_mae[functional].mean())

        print(f"\n  Noise robustness evaluation ({len(configs)} configs, "
              f"{len(functional)} functional nodes, channels={physical_channels})")

        # Pre-compute corrupt node sets per rate
        corrupt_sets = {}
        for _, rate, _, _ in configs:
            rate_key = f"r{int(rate*100)}"
            if rate_key not in corrupt_sets:
                corrupt_sets[rate_key] = select_corrupt_nodes(
                    num_nodes, rate, functional)

        results = {
            'clean_mae_functional': clean_func_mae,
            'n_functional': int(len(functional)),
            'physical_channels': physical_channels,
            'configs': {},
        }

        for noise_type, rate, severity, label in configs:
            rate_key = f"r{int(rate*100)}"
            corrupt, healthy = corrupt_sets[rate_key]
            rng = np.random.RandomState(SEED)

            # Collect per-node MAE under noise
            all_noisy_node_mae = []
            for batch in self.test_data_loader:
                data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                data['inputs'] = inject_noise(
                    data['inputs'], noise_type, corrupt,
                    severity, physical_channels, rng)

                forward_return = self.forward(
                    data, epoch=None, iter_num=None, train=False)
                pred = forward_return['prediction'].detach().cpu()
                target = forward_return['target'].detach().cpu()
                node_mae = torch.abs(pred - target).mean(dim=(1, 3))  # (B, N)
                all_noisy_node_mae.append(node_mae)

            noisy_node_mae = torch.cat(all_noisy_node_mae, dim=0).numpy().mean(axis=0)

            noisy_func = float(noisy_node_mae[functional].mean())
            healthy_clean = float(clean_node_mae[healthy].mean())
            healthy_noisy = float(noisy_node_mae[healthy].mean())
            corrupt_clean = float(clean_node_mae[corrupt].mean())
            corrupt_noisy = float(noisy_node_mae[corrupt].mean())

            cfg_result = {
                'noise_type': noise_type, 'rate': rate, 'severity': severity,
                'n_corrupt': int(len(corrupt)),
                'functional_noisy_mae': noisy_func,
                'functional_degradation_pct': (noisy_func - clean_func_mae) / clean_func_mae * 100,
                'healthy_noisy_mae': healthy_noisy,
                'healthy_degradation_pct': (healthy_noisy - healthy_clean) / max(healthy_clean, 1e-8) * 100,
                'corrupted_noisy_mae': corrupt_noisy,
                'corrupted_degradation_pct': (corrupt_noisy - corrupt_clean) / max(corrupt_clean, 1e-8) * 100,
            }
            results['configs'][label] = cfg_result

            print(f"    [{label}] func_mae={noisy_func:.2f} "
                  f"(degrad={cfg_result['functional_degradation_pct']:+.1f}%, "
                  f"spillover={cfg_result['healthy_degradation_pct']:+.1f}%)")

        return results

    @torch.no_grad()
    @master_only
    def inference(self, save_result_path: str = '') -> tuple:
        """Inference process.
        
        Args:
            save_result_path (str): The path to save the inference results. Defaults to '' meaning no saving.
        """

        data = next(iter(self.inference_dataset_loader))

        forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
        prediction = forward_return['prediction'].detach().cpu()
        prediction = prediction.squeeze(0).squeeze(-1)

        datetime_data = self._inference_get_data_time(prediction, self.inference_dataset.last_datetime, \
                                                   self.inference_dataset.description['frequency (minutes)'])

        # save
        if save_result_path:
            # save prediction to save_result_path with csv format
            datetime_data = np.fromiter((x.astype('datetime64[us]').item().strftime('%Y-%m-%d %H:%M:%S')\
                                         for x in datetime_data), 'S32').reshape(-1, 1)
            save_data = np.concatenate((datetime_data, prediction.numpy().astype(str)), axis=1)
            np.savetxt(save_result_path, save_data, delimiter=',', fmt='%s')

        return prediction.numpy(), datetime_data

    @torch.no_grad()
    @master_only
    def _inference_get_data_time(self, data: np.ndarray, last_datatime: np.datetime64, freq: int) -> np.ndarray:
        """
        Append new data to the existing data
        """

        datetime_data = np.arange(last_datatime + np.timedelta64(freq, 'm'),
                             last_datatime + np.timedelta64(freq * (len(data) + 1), 'm'),
                             np.timedelta64(freq, 'm'))

        return datetime_data

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of the validation process.

        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
        """
        greater_best = not self.metrics_best == 'min'
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/' + self.target_metrics, greater_best=greater_best)

    @master_only
    def _save_test_results(self, batch_idx: int, batch_data: Dict[str, np.ndarray]) -> None:

        """
        Save the test results to disk.
        
        Args:
            batch_idx (int): The index of the current batch.
            batch_data (Dict[np.ndarray]): The test results:{
                'inputs': np.ndarray,
                'prediction': np.ndarray,
                'target': np.ndarray,
            }
        """

        total_samples = len(self.test_data_loader.dataset)

        save_dir = os.path.join(self.ckpt_save_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)
        inputs_path = os.path.join(save_dir, 'inputs.npy')
        pred_path = os.path.join(save_dir, 'predictions.npy')
        tgt_path = os.path.join(save_dir, 'targets.npy')

        # create memmap files
        if batch_idx == 0:

            self._inputs_memmap = np.memmap(inputs_path, dtype=batch_data['inputs'].dtype, mode='w+',
                                    shape=(total_samples, *batch_data['inputs'].shape[1:]))
            self._prediction_memmap = np.memmap(pred_path, dtype=batch_data['prediction'].dtype, mode='w+',
                                    shape=(total_samples, *batch_data['prediction'].shape[1:]))
            self._target_memmap = np.memmap(tgt_path, dtype=batch_data['target'].dtype, mode='w+',
                                shape=(total_samples, *batch_data['target'].shape[1:]))

        start = batch_idx * batch_data['inputs'].shape[0]
        end = start + batch_data['inputs'].shape[0]

        self._inputs_memmap[start:end] = batch_data['inputs']
        self._prediction_memmap[start:end] = batch_data['prediction']
        self._target_memmap[start:end] = batch_data['target']

        if batch_idx == (total_samples // batch_data['inputs'].shape[0]):
            self._inputs_memmap.flush()
            self._prediction_memmap.flush()
            self._target_memmap.flush()

    def _get_metric_weight(self, x: torch.Tensor) -> int:
        """
        Get the weight for calculating metrics.
        Since the number of valid values in each batch may vary, it is necessary to perform a weighted average based on the valid value count.
        The valid value count is the total count minus the number of missing values.
        """

        if self.null_val == np.nan:
            valid_num = (~torch.isnan(x)).sum().item()
        else:
            eps = 5e-5
            valid_num = (~torch.isclose(x, torch.tensor(self.null_val).expand_as(x).to(x.device), atol=eps, rtol=0.0)).sum().item()

        return valid_num
