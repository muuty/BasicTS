import torch
import torch.nn as nn
import numpy as np

from .model import STAEformer
from .spillover_corrector import SpilloverCorrector


# --- Noise injection functions (self-contained to avoid import from experiments/) ---

def _apply_gaussian_noise(inputs, corrupt_nodes, severity, physical_channels, rng):
    """Additive Gaussian noise on physical channels. severity = noise_std / channel_std."""
    corrupted = inputs.clone()
    B, T = corrupted.shape[:2]
    for ch in physical_channels:
        ch_std = corrupted[:, :, :, ch].std().item()
        noise_std = severity * ch_std
        noise = torch.tensor(
            rng.normal(0, noise_std, (B, T, len(corrupt_nodes))),
            dtype=corrupted.dtype, device=corrupted.device,
        )
        corrupted[:, :, corrupt_nodes, ch] += noise
    for ch in physical_channels:
        corrupted[:, :, corrupt_nodes, ch].clamp_(min=0)
    return corrupted


class STAEformerRobust(nn.Module):
    """STAEformer with noise augmentation training and optional SpilloverCorrector.

    During training:
      1. Clean forward  → μ_c (+ δ_c if corrector enabled)
      2. Noise injection into random nodes
      3. Noisy forward  → μ_n (+ δ_n if corrector enabled)
      4. Returns both predictions + corrupt_mask for loss computation

    During inference:
      Single forward → μ + δ (or just μ if no corrector)

    Set use_corrector=False for noise-augmentation-only baseline (Exp 1).
    """

    def __init__(
        self,
        use_corrector=True,
        corrupt_ratio=0.1,
        noise_severity=0.5,
        physical_channels=(0, 1, 2),
        seed=42,
        backbone_ckpt=None,
        freeze_backbone=False,
        **backbone_params,
    ):
        super().__init__()

        # Extract corrector params before passing to backbone
        d_model = (
            backbone_params.get('input_embedding_dim', 24)
            + backbone_params.get('tod_embedding_dim', 24)
            + backbone_params.get('dow_embedding_dim', 24)
            + backbone_params.get('spatial_embedding_dim', 0)
            + backbone_params.get('adaptive_embedding_dim', 80)
        )

        self.backbone = STAEformer(**backbone_params)

        # Sequential training: load pre-trained backbone and freeze
        if backbone_ckpt:
            ckpt = torch.load(backbone_ckpt, map_location='cpu')
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.backbone.load_state_dict(sd)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.use_corrector = use_corrector
        if use_corrector:
            self.corrector = SpilloverCorrector(
                d_model=d_model,
                n_heads=backbone_params.get('num_heads', 4),
                dropout=backbone_params.get('dropout', 0.1),
            )

        self.corrupt_ratio = corrupt_ratio
        self.noise_severity = noise_severity
        self.physical_channels = list(physical_channels)
        self.rng = np.random.RandomState(seed)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _forward_backbone(self, history_data):
        """Run backbone encoder → spatial → decoder, return prediction + hidden."""
        if self.freeze_backbone:
            with torch.no_grad():
                x, graph = self.backbone.encoder(history_data)
                x = self.backbone.spatial(x, graph)
                mu = self.backbone.decoder(x)
            return mu, x
        x, graph = self.backbone.encoder(history_data)
        x = self.backbone.spatial(x, graph)
        mu = self.backbone.decoder(x)        # (B, T_out, N, 1)
        return mu, x                          # x = hidden (B, T_in, N, d_model)

    def _apply_corrector(self, hidden, mu):
        """Apply SpilloverCorrector to hidden features, return corrected prediction."""
        h = hidden.detach().mean(dim=1)       # (B, N, d_model) — stop gradient + temporal pool
        delta, reliability = self.corrector(h)  # (B, N, 1) each
        # broadcast delta to match prediction shape: (B, N, 1) → (B, T_out, N, 1)
        delta_broadcast = delta.unsqueeze(1).expand_as(mu)
        return mu + delta_broadcast, delta, reliability

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        # --- Clean forward ---
        mu_c, hidden_c = self._forward_backbone(history_data)

        if self.use_corrector:
            pred_c, delta_c, rel_c = self._apply_corrector(hidden_c, mu_c)
        else:
            pred_c = mu_c

        if not train:
            return {'prediction': pred_c}

        # --- Training: noise injection + noisy forward ---
        N = history_data.shape[2]
        K = max(1, int(N * self.corrupt_ratio))
        corrupt_nodes = self.rng.choice(N, size=K, replace=False)

        noisy_input = _apply_gaussian_noise(
            history_data, corrupt_nodes, self.noise_severity,
            self.physical_channels, self.rng,
        )

        mu_n, hidden_n = self._forward_backbone(noisy_input)

        if self.use_corrector:
            pred_n, delta_n, rel_n = self._apply_corrector(hidden_n, mu_n)
        else:
            pred_n = mu_n

        # Build corrupt mask: (B, N) boolean — True for corrupted nodes
        B = history_data.shape[0]
        corrupt_mask = torch.zeros(B, N, dtype=torch.bool, device=history_data.device)
        corrupt_mask[:, corrupt_nodes] = True

        return {
            'prediction': pred_c,              # clean prediction (used by runner for metrics)
            'prediction_noisy': pred_n,        # noisy prediction (for L_noisy)
            'corrupt_mask': corrupt_mask,      # (B, N) which nodes were corrupted
        }
