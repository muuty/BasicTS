"""
Disentangled Temporal Pretrain Runner.

Handles pretraining of DisentangledTemporalModel with:
- Temporal-Aware Contrastive Loss (on z_context)
- Reconstruction Loss (on z_self)

Config:
    CFG.PRETRAIN = {
        'contrastive_weight': 1.0,
        'reconstruction_weight': 1.0,
    }
"""
import torch
from basicts.runners import SimpleTimeSeriesForecastingRunner


class DisentangledTemporalRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for disentangled temporal pretraining (contrastive + reconstruction)."""

    def __init__(self, cfg: dict):
        self._pretrain_cfg = cfg.get('PRETRAIN', {})
        self._contrastive_weight = self._pretrain_cfg.get('contrastive_weight', 1.0)
        self._reconstruction_weight = self._pretrain_cfg.get('reconstruction_weight', 1.0)
        self._meters_registered = False

        super().__init__(cfg)

        # Override loss function
        self.loss = self._compute_loss

    def _compute_loss(self, prediction, target, loss=None, **kwargs):
        """Dummy loss - actual loss computed in forward.

        Note: 'loss' must be an explicit parameter (not just **kwargs) because
        metric_forward() filters args based on function signature.
        """
        if loss is not None:
            return loss
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    def on_epoch_start(self, epoch: int) -> None:
        """Register meters on first epoch."""
        super().on_epoch_start(epoch)
        if not self._meters_registered:
            self.register_epoch_meter('train/loss_contrastive', 'train', '{:.4f}')
            self.register_epoch_meter('train/loss_reconstruction', 'train', '{:.4f}')
            self._meters_registered = True

    def forward(self, data, epoch=None, iter_num=None, train=True, **kwargs):
        """Forward pass with disentangled temporal pretraining losses."""
        data = self.preprocessing(data)
        x = self.select_input_features(self.to_running_device(data['inputs']))

        self.model.train() if train else self.model.eval()
        output = self.model(x)

        # Contrastive loss on z_context
        contrastive_loss = self.model.compute_contrastive_loss(
            output['z1'],
            output['z2'],
            output.get('tod'),
            output.get('dow'),
        )

        # Reconstruction loss on z_self
        reconstruction_loss = self.model.compute_reconstruction_loss(
            output['x_original'],
            output['x_recon'],
            output['mask'],
        )

        # Total loss
        total_loss = (
            self._contrastive_weight * contrastive_loss +
            self._reconstruction_weight * reconstruction_loss
        )

        # Update meters
        if train:
            self.update_epoch_meter('train/loss_contrastive', contrastive_loss.item())
            self.update_epoch_meter('train/loss_reconstruction', reconstruction_loss.item())

        return {
            'prediction': output['x_recon'],
            'target': output['x_original'],
            'loss': total_loss
        }
