import torch
import torch.nn as nn
import torch.nn.functional as F

from .staeformer_arch import STAEformer


class DecomposedSTAEformer(nn.Module):
    """Decomposed prediction: y = mu_y + sigma_y * r_y

    Scale head predicts future per-node mean and std (mu_y, sigma_y).
    Backbone (STAEformer) predicts normalized residual (r_y) from instance-normed input.

    This architectural decomposition separates scale prediction from pattern prediction,
    making each component independently more robust to distribution shift.
    """

    def __init__(self, backbone_params, scale_head_hidden=32):
        super().__init__()
        self.backbone = STAEformer(**backbone_params)
        # Scale head: (mu_x, sigma_x, last_flow, tod, dow) -> (mu_y_hat, log_sigma_y_hat)
        self.scale_head = nn.Sequential(
            nn.Linear(5, scale_head_hidden),
            nn.ReLU(),
            nn.Linear(scale_head_hidden, 2),
        )

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        # history_data: (B, T_in, N, C) in Z-score space
        flow_z = history_data[:, :, :, 0]  # (B, T, N)

        # Per-node input statistics (Z-score space)
        mu_x = flow_z.mean(dim=1)           # (B, N)
        sigma_x = flow_z.std(dim=1) + 1e-5  # (B, N)
        last_flow = flow_z[:, -1, :]         # (B, N)

        # Time features from last input step
        tod = history_data[:, -1, :, -2]  # (B, N)
        dow = history_data[:, -1, :, -1]  # (B, N)

        # Scale head: predict future mu_y, sigma_y
        scale_input = torch.stack([mu_x, sigma_x, last_flow, tod, dow], dim=-1)  # (B, N, 5)
        scale_out = self.scale_head(scale_input)  # (B, N, 2)
        mu_y_hat = scale_out[:, :, 0]                      # (B, N)
        sigma_y_hat = F.softplus(scale_out[:, :, 1]) + 1e-5  # (B, N), positive

        # Instance normalize input for backbone
        history_normed = history_data.clone()
        history_normed[:, :, :, 0] = (flow_z - mu_x.unsqueeze(1)) / sigma_x.unsqueeze(1)

        # Backbone predicts normalized residual
        backbone_out = self.backbone(history_normed, future_data, batch_seen, epoch, train)
        r_y_hat = backbone_out['prediction']  # (B, T_out, N, 1)

        # Decomposed prediction: y = mu_y + sigma_y * r_y
        mu = mu_y_hat.unsqueeze(1).unsqueeze(-1)     # (B, 1, N, 1)
        sigma = sigma_y_hat.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        prediction = mu + sigma * r_y_hat  # (B, T_out, N, 1)

        return {
            'prediction': prediction,
            'mu_y_hat': mu_y_hat,
            'sigma_y_hat': sigma_y_hat,
        }
