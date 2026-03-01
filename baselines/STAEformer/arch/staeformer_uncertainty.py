import torch
import torch.nn as nn
import torch.nn.functional as F

from .staeformer_arch import STAEformer


class STAEformerUncertainty(nn.Module):
    """STAEformer with uncertainty output (mu, sigma).

    Outputs point prediction (mu) and aleatoric uncertainty (sigma) per node per timestep.
    sigma is learned via heteroscedastic NLL loss — nodes with noisy/unreliable input
    automatically get higher sigma (lower confidence).

    Architecture: shared STAEformer backbone with output_dim=2,
    split into mu (prediction) and raw_sigma (-> softplus -> sigma).
    """

    def __init__(self, sigma_min=1e-3, **backbone_params):
        super().__init__()
        # Force output_dim=2 for (mu, raw_sigma)
        backbone_params['output_dim'] = 2
        self.backbone = STAEformer(**backbone_params)
        self.sigma_min = sigma_min

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        out = self.backbone(history_data, future_data, batch_seen, epoch, train, **kwargs)
        raw = out['prediction']  # (B, T_out, N, 2)

        mu = raw[..., 0:1]       # (B, T_out, N, 1)
        raw_sigma = raw[..., 1:2]  # (B, T_out, N, 1)
        sigma = F.softplus(raw_sigma) + self.sigma_min  # positive, bounded below

        return {
            'prediction': mu,
            'sigma': sigma,
            'epoch': torch.tensor(epoch if epoch is not None else 0, device=mu.device),
        }
