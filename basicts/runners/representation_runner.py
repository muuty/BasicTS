"""
Representation Runner for two-stage training:
  - Stage 1 (pretrain): Contrastive pre-training of encoder
  - Stage 2 (prediction): Prediction training with frozen encoder + backbone

Reference: design.md - Section 3.5
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class RepresentationRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner supporting two-stage training:

    Stage 1 (pretrain):
        - Train encoder with contrastive loss
        - Uses weak and strong augmentation
        - Output: pre-trained encoder checkpoint

    Stage 2 (prediction):
        - Load pre-trained encoder (frozen)
        - Train backbone with prediction loss
        - Output: full model checkpoint

    Config keys:
        TRAIN.STAGE: 'pretrain' or 'prediction'

        For pretrain:
            PRETRAIN.AUGMENTATION: augmentation config
            PRETRAIN.CONTRASTIVE_LOSS: loss config

        For prediction:
            TRAIN.ENCODER_CKPT: path to pre-trained encoder
            TRAIN.FREEZE_ENCODER: whether to freeze encoder (default: True)
    """

    def __init__(self, cfg: Dict):
        # Get training stage before super().__init__
        self.training_stage = cfg['TRAIN'].get('STAGE', 'prediction')

        super().__init__(cfg)

        # Stage-specific initialization
        if self.training_stage == 'pretrain':
            self._init_pretrain_components(cfg)
        else:
            self._init_prediction_components(cfg)

    def _init_pretrain_components(self, cfg: Dict):
        """Initialize components for pre-training stage."""
        pretrain_cfg = cfg.get('PRETRAIN', {})

        # Augmentation
        self.weak_aug_std = pretrain_cfg.get('WEAK_AUG_STD', 0.1)
        self.strong_aug_cfg = pretrain_cfg.get('STRONG_AUGMENTATION', None)

        # Initialize augmentor if configured
        if self.strong_aug_cfg is not None:
            self._build_augmentor(self.strong_aug_cfg)
        else:
            self.augmentor = None

        # Contrastive loss
        loss_cfg = pretrain_cfg.get('CONTRASTIVE_LOSS', {})
        self._build_contrastive_loss(loss_cfg)

        # Register meters for pre-training
        self.register_epoch_meter('train/contrastive_loss', 'train', '{:.4f}')

        self.logger.info(f"RepresentationRunner initialized in PRETRAIN stage")
        self.logger.info(f"  Weak augmentation std: {self.weak_aug_std}")
        self.logger.info(f"  Strong augmentation: {self.strong_aug_cfg is not None}")

    def _init_prediction_components(self, cfg: Dict):
        """Initialize components for prediction stage."""
        # Encoder checkpoint loading is handled by model building
        self.freeze_encoder = cfg['TRAIN'].get('FREEZE_ENCODER', True)

        self.logger.info(f"RepresentationRunner initialized in PREDICTION stage")
        self.logger.info(f"  Freeze encoder: {self.freeze_encoder}")

    def _build_augmentor(self, aug_cfg: Dict):
        """Build augmentor for strong augmentation."""
        aug_type = aug_cfg.get('TYPE', 'TrafficAnomalyInjector')

        if aug_type == 'TrafficAnomalyInjector':
            from contrastive.augmentation import TrafficAnomalyInjector
            self.augmentor = TrafficAnomalyInjector(
                severity_levels=aug_cfg.get('SEVERITY_LEVELS', {
                    1: (0.3, 0.5),
                    2: (0.5, 0.8),
                    3: (0.8, 1.0),
                }),
                propagation_hops=aug_cfg.get('PROPAGATION_HOPS', 2),
                propagation_decay=aug_cfg.get('PROPAGATION_DECAY', 0.5),
                recovery_steps=aug_cfg.get('RECOVERY_STEPS', 6),
            )
        else:
            raise ValueError(f"Unknown augmentor type: {aug_type}")

    def _build_contrastive_loss(self, loss_cfg: Dict):
        """Build contrastive loss function."""
        loss_type = loss_cfg.get('TYPE', 'SeverityAwareContrastiveLoss')

        if loss_type == 'SeverityAwareContrastiveLoss':
            from contrastive.contrastive_loss import SeverityAwareContrastiveLoss
            self.contrastive_loss_fn = SeverityAwareContrastiveLoss(
                temperature=loss_cfg.get('TEMPERATURE', 0.1),
                severity_weight_scale=loss_cfg.get('SEVERITY_WEIGHT_SCALE', 1.0),
            )
        elif loss_type in ('InfoNCE', 'InfoNCELoss'):
            from contrastive.contrastive_loss import InfoNCELoss
            self.contrastive_loss_fn = InfoNCELoss(
                temperature=loss_cfg.get('TEMPERATURE', 0.1),
            )
        else:
            raise ValueError(f"Unknown contrastive loss type: {loss_type}")

    def weak_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weak augmentation (Gaussian jittering)."""
        noise = torch.randn_like(x) * self.weak_aug_std
        return x + noise

    def strong_augmentation(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None):
        """
        Apply strong augmentation (anomaly injection).

        Returns:
            x_aug: Augmented data
            severity_labels: Severity labels for each position
        """
        if self.augmentor is None:
            # Fallback to stronger jittering
            noise = torch.randn_like(x) * (self.weak_aug_std * 3)
            severity_labels = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device)
            return x + noise, severity_labels

        return self.augmentor.inject(x, adj_matrix)

    def train_iters(self, epoch: int, iter_index: int, data: Dict) -> torch.Tensor:
        """
        Training iteration for either pretrain or prediction stage.
        """
        if self.training_stage == 'pretrain':
            return self._train_pretrain_iter(epoch, iter_index, data)
        else:
            return self._train_prediction_iter(epoch, iter_index, data)

    def _train_pretrain_iter(self, epoch: int, iter_index: int, data: Dict) -> torch.Tensor:
        """
        Pre-training iteration with contrastive loss.
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index

        # Preprocess
        data = self.preprocessing(data)
        history_data = data['inputs']
        history_data = self.to_running_device(history_data)

        # Get context indices if available
        time_of_day_idx = data.get('time_of_day_idx', None)
        day_of_week_idx = data.get('day_of_week_idx', None)

        if time_of_day_idx is not None:
            time_of_day_idx = self.to_running_device(time_of_day_idx)
        if day_of_week_idx is not None:
            day_of_week_idx = self.to_running_device(day_of_week_idx)

        # Weak augmentation
        x_weak = self.weak_augmentation(history_data)
        severity_weak = torch.zeros(
            x_weak.shape[0], x_weak.shape[1], x_weak.shape[2],
            device=x_weak.device
        )

        # Strong augmentation
        adj_matrix = getattr(self, 'adj_matrix', None)
        x_strong, severity_strong = self.strong_augmentation(history_data, adj_matrix)

        # Forward pass for both views
        # Model should be ContextAwareSTEncoder
        z_weak = self.model(
            x_weak,
            time_of_day_idx=time_of_day_idx,
            day_of_week_idx=day_of_week_idx,
        )
        z_strong = self.model(
            x_strong,
            time_of_day_idx=time_of_day_idx,
            day_of_week_idx=day_of_week_idx,
        )

        # Contrastive loss
        loss = self.contrastive_loss_fn(z_weak, z_strong, severity_weak, severity_strong)

        # Update meters
        self.update_epoch_meter('train/contrastive_loss', loss.item())

        return loss

    def _train_prediction_iter(self, epoch: int, iter_index: int, data: Dict) -> torch.Tensor:
        """
        Prediction training iteration (standard forecasting loss).
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # Prediction loss
        loss = self.metric_forward(self.loss, forward_return)

        # Update meters
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return loss

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None,
                train: bool = True, **kwargs) -> Dict:
        """
        Forward pass.

        For pretrain stage: returns encoder output
        For prediction stage: returns prediction output
        """
        if self.training_stage == 'pretrain':
            return self._forward_pretrain(data, epoch, iter_num, train)
        else:
            return self._forward_prediction(data, epoch, iter_num, train)

    def _forward_pretrain(self, data: Dict, epoch: int, iter_num: int, train: bool) -> Dict:
        """Forward for pre-training (encoder only)."""
        data = self.preprocessing(data)
        history_data = data['inputs']
        history_data = self.to_running_device(history_data)

        # Get context indices
        time_of_day_idx = data.get('time_of_day_idx', None)
        day_of_week_idx = data.get('day_of_week_idx', None)

        if time_of_day_idx is not None:
            time_of_day_idx = self.to_running_device(time_of_day_idx)
        if day_of_week_idx is not None:
            day_of_week_idx = self.to_running_device(day_of_week_idx)

        # Encoder forward
        z = self.model(
            history_data,
            time_of_day_idx=time_of_day_idx,
            day_of_week_idx=day_of_week_idx,
        )

        return {
            'representation': z,
            'inputs': history_data,
        }

    def _forward_prediction(self, data: Dict, epoch: int, iter_num: int, train: bool) -> Dict:
        """Forward for prediction (full model)."""
        # Use parent class forward
        return super().forward(data, epoch, iter_num, train)

    def save_model(self, epoch: int):
        """Save model checkpoint."""
        super().save_model(epoch)

        # For pretrain stage, also save encoder separately
        if self.training_stage == 'pretrain':
            encoder_path = os.path.join(self.ckpt_save_dir, f'encoder_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
            }, encoder_path)
            self.logger.info(f"Encoder checkpoint saved: {encoder_path}")

    def save_best_model(self, epoch: int, metric_name: str = None):
        """Save best model checkpoint."""
        super().save_best_model(epoch, metric_name)

        # For pretrain stage, also save best encoder
        if self.training_stage == 'pretrain':
            encoder_path = os.path.join(self.ckpt_save_dir, 'encoder_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
            }, encoder_path)
            self.logger.info(f"Best encoder checkpoint saved: {encoder_path}")

    def test(self, train_epoch: Optional[int] = None,
             save_metrics: bool = False, save_results: bool = False) -> Dict:
        """
        Test process.

        For pretrain: skip testing (or do representation quality evaluation)
        For prediction: standard testing
        """
        if self.training_stage == 'pretrain':
            self.logger.info("Pre-training stage: skipping standard test")
            return {}
        else:
            return super().test(train_epoch, save_metrics, save_results)
