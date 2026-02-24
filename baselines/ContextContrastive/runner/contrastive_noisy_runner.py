"""
Contrastive Noisy Runner: Noise-Invariant Representation Learning.

Trains encoder to produce representations that are invariant to input noise.
During training, both clean and noisy versions are encoded, and an alignment
loss enforces ||encoder(x_noisy) - encoder(x_clean)||² ≈ 0.

Total loss = L_pred(y_hat, y) + λ * L_align(z_clean, z_noisy)

Theoretical justification: If encoder maps noisy inputs to the same
representation as clean inputs, the downstream predictor becomes
inherently robust to sensor noise at test time.
"""

from typing import Dict, Optional, Union, Tuple

import torch

from .noisy_representation_learning_runner import NoisyRepresentationLearningRunner


class ContrastiveNoisyRunner(NoisyRepresentationLearningRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        contrastive_cfg = cfg.get('CONTRASTIVE', {})
        self.alignment_lambda = contrastive_cfg.get('lambda', 0.1)
        self._alignment_loss = None

    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:

        if not train:
            return super().forward(data, epoch, iter_num, train, **kwargs)

        # === Training: dual encoding for alignment loss ===

        target_mask = None
        if self.mask_channel_idx is not None:
            target_mask = data['target'][:, :, :, self.mask_channel_idx:self.mask_channel_idx + 1].clone()

        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        # 1. Save clean version
        history_clean = history_data.clone()

        # 2. Inject noise
        history_noisy = self._inject_noise(history_data)

        # 3. Encode both (encoder must be trainable)
        self.encoder.train()

        z_clean = self.encoder.encode(history_clean)
        z_noisy = self.encoder.encode(history_noisy)

        # 4. Alignment loss: MSE between clean and noisy representations
        # Only on physical channels (not tod/dow)
        self._alignment_loss = torch.nn.functional.mse_loss(z_noisy, z_clean)

        # 5. Predict from noisy encoding (downstream model sees noisy-encoded input)
        encoded_history = self._add_tod_dow_if_needed(z_noisy, history_noisy)

        if self.downstream_features is not None:
            encoded_history = encoded_history[..., self.downstream_features]

        model_return = self.model(
            history_data=encoded_history,
            future_data=future_data_4_dec,
            batch_seen=iter_num, epoch=epoch, train=train
        )

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_noisy)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes]

        model_return = self.postprocessing(model_return)
        return model_return

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        pred_loss = self.metric_forward(self.loss, forward_return)

        # Combine: prediction loss + alignment loss
        align_loss = self._alignment_loss if self._alignment_loss is not None else 0.0
        loss = pred_loss + self.alignment_lambda * align_loss

        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)

        return loss
