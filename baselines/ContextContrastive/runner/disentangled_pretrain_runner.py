"""
Disentangled Pretrain Runner.

Handles pretraining of DisentangledPretrainModel with:
- Reconstruction loss (MSE on masked positions)
- Optional orthogonality loss

Config:
    CFG.PRETRAIN = {
        'recon_weight': 1.0,
        'ortho_weight': 0.1,  # Only used if model.use_orthogonality=True
    }
"""
import torch
import torch.nn.functional as F
from basicts.runners import SimpleTimeSeriesForecastingRunner


class DisentangledPretrainRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for disentangled representation pretraining."""

    def __init__(self, cfg: dict):
        self._pretrain_cfg = cfg.get('PRETRAIN', {})
        self._recon_weight = self._pretrain_cfg.get('recon_weight', 1.0)
        self._meters_registered = False

        super().__init__(cfg)

        # Override loss function
        self.loss = self._compute_loss

    def _compute_loss(self, prediction, target, **kwargs):
        """Dummy loss - actual loss computed in forward."""
        return kwargs.get('loss', torch.tensor(0.0))

    def on_epoch_start(self, epoch: int) -> None:
        """Register meters on first epoch."""
        super().on_epoch_start(epoch)
        if not self._meters_registered:
            self.register_epoch_meter('train/loss_recon', 'train', '{:.4f}')
            self.register_epoch_meter('train/loss_ortho', 'train', '{:.4f}')
            self.register_epoch_meter('train/loss_recon_context', 'train', '{:.4f}')
            self.register_epoch_meter('train/loss_recon_self', 'train', '{:.4f}')
            self._meters_registered = True

    def forward(self, data, epoch=None, iter_num=None, train=True, **kwargs):
        """Forward pass with disentangled pretraining losses."""
        data = self.preprocessing(data)
        x = self.select_input_features(self.to_running_device(data['inputs']))

        self.model.train() if train else self.model.eval()
        output = self.model(x, return_all=True)

        # Reconstruction loss (only on masked positions)
        mask = output['mask']
        x_original = output['x_original']

        # Main reconstruction loss (from fused representation)
        x_recon = output['x_recon']
        recon_loss = F.mse_loss(x_recon[mask], x_original[mask])

        # Individual reconstruction losses (for monitoring)
        x_recon_context = output.get('x_recon_context')
        x_recon_self = output.get('x_recon_self')

        recon_loss_context = F.mse_loss(x_recon_context[mask], x_original[mask]) if x_recon_context is not None else torch.tensor(0.0)
        recon_loss_self = F.mse_loss(x_recon_self[mask], x_original[mask]) if x_recon_self is not None else torch.tensor(0.0)

        # Orthogonality loss (handle case when it doesn't require grad)
        ortho_loss = output.get('ortho_loss')
        if ortho_loss is None or not ortho_loss.requires_grad:
            ortho_loss_value = 0.0 if ortho_loss is None else ortho_loss.item()
            ortho_loss = torch.zeros_like(recon_loss)  # Match dtype/device, has grad_fn from recon_loss context
        else:
            ortho_loss_value = ortho_loss.item()

        # Total loss
        total_loss = self._recon_weight * recon_loss + ortho_loss

        # Update meters
        if train:
            self.update_epoch_meter('train/loss_recon', recon_loss.item())
            self.update_epoch_meter('train/loss_ortho', ortho_loss_value if isinstance(ortho_loss_value, float) else ortho_loss.item())
            self.update_epoch_meter('train/loss_recon_context', recon_loss_context.item())
            self.update_epoch_meter('train/loss_recon_self', recon_loss_self.item())

        return {
            'prediction': x_recon,
            'target': x_original,
            'loss': total_loss
        }
