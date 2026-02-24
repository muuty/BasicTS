"""
Unified Pretrain Runner.

Config:
    CFG.PRETRAIN = {
        'LOSS': {
            'contrastive': {'enabled': True, 'weight': 1.0, 'temperature': 0.1},
            'reconstruction': {'enabled': True, 'weight': 0.5, 'target': 'masked'},
        },
    }
    # adj_path goes in MODEL.PARAM, not PRETRAIN
"""
import torch
from basicts.runners import SimpleTimeSeriesForecastingRunner
from ..loss.combined_loss import CombinedPretrainLoss


class PretrainRunner(SimpleTimeSeriesForecastingRunner):
    """Unified runner for pretraining (contrastive, reconstruction, or hybrid)."""

    def __init__(self, cfg: dict):
        pretrain_cfg = cfg.get('PRETRAIN', {})
        self.pretrain_loss = CombinedPretrainLoss(pretrain_cfg.get('LOSS', {'contrastive': {'enabled': True}}))
        self._pretrain_loss_cfg = pretrain_cfg.get('LOSS', {})

        super().__init__(cfg)

        # Use pre-computed loss from forward
        self.loss = lambda prediction, target, loss, **kwargs: loss

        self._log_pretrain_config(pretrain_cfg)

    def on_epoch_start(self, epoch: int) -> None:
        """Register pretrain loss meters (only on first epoch)."""
        super().on_epoch_start(epoch)
        # Register meters only on first epoch
        if epoch == 1:
            if self._pretrain_loss_cfg.get('contrastive', {}).get('enabled'):
                self.register_epoch_meter('train/loss_contrastive', 'train', '{:.4f}')
            if self._pretrain_loss_cfg.get('reconstruction', {}).get('enabled'):
                self.register_epoch_meter('train/loss_reconstruction', 'train', '{:.4f}')

    def _log_pretrain_config(self, cfg):
        self.logger.info("=" * 50)
        self.logger.info("PRETRAIN CONFIG")
        loss_cfg = cfg.get('LOSS', {})
        if loss_cfg.get('contrastive', {}).get('enabled'):
            c = loss_cfg['contrastive']
            self.logger.info(f"  Contrastive: weight={c.get('weight', 1.0)}, temp={c.get('temperature', 0.1)}")
        if loss_cfg.get('reconstruction', {}).get('enabled'):
            r = loss_cfg['reconstruction']
            self.logger.info(f"  Reconstruction: weight={r.get('weight', 0.5)}, target={r.get('target', 'masked')}")
        self.logger.info("=" * 50)

    def forward(self, data, epoch=None, iter_num=None, train=True, **kwargs):
        data = self.preprocessing(data)
        history_data = self.select_input_features(self.to_running_device(data['inputs']))

        self.model.train() if train else self.model.eval()
        output = self.model(history_data, return_reconstruction=True)

        loss, loss_dict = self.pretrain_loss(
            z1=output.get('z1'), z2=output.get('z2'),
            x_original=output.get('x_original'), x_reconstructed=output.get('x_recon'), mask=output.get('mask'))

        for k, v in loss_dict.items():
            if k != 'total':
                self.update_epoch_meter(f'train/loss_{k}', v)

        x_recon = output.get('x_recon')
        prediction = x_recon if x_recon is not None else history_data
        return {'prediction': prediction, 'target': history_data, 'loss': loss}
