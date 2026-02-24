"""
Predictive Contrastive Pretrain Runner.

Handles pretraining of PredictiveContrastiveModel:
- Encodes past and future halves of input
- Computes cosine similarity loss between predicted and actual future representations
"""
import torch
from basicts.runners import SimpleTimeSeriesForecastingRunner


class PredictiveContrastiveRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for predictive contrastive pretraining."""

    def __init__(self, cfg: dict):
        self._meters_registered = False
        super().__init__(cfg)
        self.loss = self._compute_loss

    def _compute_loss(self, prediction, target, loss=None, **kwargs):
        """Dummy loss - actual loss computed in forward."""
        if loss is not None:
            return loss
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    def on_epoch_start(self, epoch: int) -> None:
        super().on_epoch_start(epoch)
        if not self._meters_registered:
            self.register_epoch_meter('train/loss_predictive', 'train', '{:.4f}')
            self._meters_registered = True

    def forward(self, data, epoch=None, iter_num=None, train=True, **kwargs):
        data = self.preprocessing(data)
        x = self.select_input_features(self.to_running_device(data['inputs']))

        self.model.train() if train else self.model.eval()
        output = self.model(x)

        loss = self.model.compute_loss(output['z_pred'], output['z_target'])

        if train:
            self.update_epoch_meter('train/loss_predictive', loss.item())

        # Return dummy prediction/target for metric computation
        B, T, N, C = x.shape
        dummy = x[:, :1, :, :1].detach()
        return {
            'prediction': dummy,
            'target': dummy,
            'loss': loss,
        }
