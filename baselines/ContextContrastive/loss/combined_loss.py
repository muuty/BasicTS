"""
Combined Loss for Pretraining.

Config:
    loss_config = {
        'contrastive': {'enabled': True, 'weight': 1.0, 'temperature': 0.1},
        'reconstruction': {'enabled': True, 'weight': 0.5, 'target': 'masked'},
    }
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedPretrainLoss(nn.Module):
    """Unified loss for pretraining (contrastive, reconstruction, or hybrid)."""

    def __init__(self, loss_config: dict):
        super().__init__()
        # Contrastive settings
        contrastive_cfg = loss_config.get('contrastive', {})
        self.use_contrastive = contrastive_cfg.get('enabled', False)
        self.contrastive_weight = contrastive_cfg.get('weight', 1.0)
        self.temperature = contrastive_cfg.get('temperature', 0.1)

        # Reconstruction settings
        recon_cfg = loss_config.get('reconstruction', {})
        self.use_reconstruction = recon_cfg.get('enabled', False)
        self.recon_weight = recon_cfg.get('weight', 0.5)
        self.recon_target = recon_cfg.get('target', 'masked')
        self.recon_loss_fn = F.l1_loss if recon_cfg.get('loss_type') == 'l1' else F.mse_loss

    def forward(self, z1=None, z2=None, x_original=None, x_reconstructed=None, mask=None):
        """Compute combined loss. Tensors must be provided based on enabled losses."""
        device = (z1 if z1 is not None else x_original).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        if self.use_contrastive:
            contrastive_loss = self._simclr_loss(z1, z2)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['contrastive'] = contrastive_loss.item()

        if self.use_reconstruction:
            recon_loss = self._reconstruction_loss(x_original, x_reconstructed, mask)
            total_loss = total_loss + self.recon_weight * recon_loss
            loss_dict['reconstruction'] = recon_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

    def _simclr_loss(self, z1, z2):
        """SimCLR-style InfoNCE loss. z1, z2: [B, N, D] or [B, D]"""
        # Flatten if 3D
        if z1.dim() == 3:
            z1 = z1.flatten(0, 1)
            z2 = z2.flatten(0, 1)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        n = z1.shape[0]

        # Similarity matrix [2n, 2n]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask diagonal
        sim.fill_diagonal_(float('-inf'))
        sim[:n, :n].fill_diagonal_(float('-inf'))
        sim[n:, n:].fill_diagonal_(float('-inf'))

        # Labels: positive pairs are (i, i+n) and (i+n, i)
        labels = torch.cat([torch.arange(n, 2*n, device=z.device),
                           torch.arange(n, device=z.device)])
        return F.cross_entropy(sim, labels)

    def _reconstruction_loss(self, x_original, x_reconstructed, mask):
        """Reconstruction loss. If target='masked', only compute on masked positions."""
        if self.recon_target == 'masked' and mask is not None and mask.any():
            mask = mask.expand_as(x_original)
            return self.recon_loss_fn(x_reconstructed[mask], x_original[mask])
        return self.recon_loss_fn(x_reconstructed, x_original)


def get_combined_loss(loss_config: dict):
    return CombinedPretrainLoss(loss_config)
