"""
Disentangled + Temporal-Aware Pretrain Model.

Combines the best of both approaches:
- z_context: Trained with Temporal-Aware Contrastive Loss
  (smart negative sampling based on tod/dow)
- z_self: Trained with Reconstruction Loss
  (direct supervision, no false negatives)

This should give:
- Better overall MAE (from temporal-aware negatives)
- Better robustness (from self-representation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .disentangled_encoder import DisentangledEncoder


class DisentangledTemporalModel(nn.Module):
    """
    Disentangled pretraining with temporal-aware contrastive + reconstruction.

    Architecture:
        Input → DisentangledEncoder → z_context, z_self
                                           ↓           ↓
                              Temporal-Aware      Reconstruction
                              Contrastive Loss       Loss
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        fusion: str = 'concat',
        mask_ratio: float = 0.5,
        # Temporal-aware contrastive params
        temperature: float = 0.1,
        tod_weight: float = 0.5,
        dow_weight: float = 0.5,
        soft_positive_weight: float = 0.3,
        # Loss weights
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        # Reconstruction target config
        recon_dim: int = None,  # If set, reconstruct only first recon_dim features
        tod_idx: int = -2,  # Index of tod in input (for extraction), -2 means second to last
        dow_idx: int = -1,  # Index of dow in input (for extraction), -1 means last
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.recon_dim = recon_dim if recon_dim is not None else input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.tod_weight = tod_weight
        self.dow_weight = dow_weight
        self.soft_positive_weight = soft_positive_weight
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.tod_idx = tod_idx
        self.dow_idx = dow_idx

        # Disentangled encoder
        self.encoder = DisentangledEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            fusion=fusion,
        )

        # Learnable mask token for reconstruction (only for traffic features)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.recon_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Projection head for contrastive learning (on z_context)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Reconstruction head (on z_self) - only reconstructs traffic features
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.recon_dim),
        )

    def apply_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial masking for reconstruction task.

        Only masks traffic features (first recon_dim features).
        Time features (tod, dow) are preserved.
        """
        B, T, N, C = x.shape

        # Create node-level mask (same mask across time)
        node_mask = torch.rand(B, 1, N, 1, device=x.device) < self.mask_ratio

        # Only mask traffic features (first recon_dim features)
        x_traffic = x[..., :self.recon_dim]  # [B, T, N, recon_dim]

        # Expand mask for traffic features only
        traffic_mask = node_mask.expand(B, T, N, self.recon_dim)
        mask_token_expanded = self.mask_token.expand(B, T, N, self.recon_dim)

        # Apply masking to traffic features
        x_traffic_masked = torch.where(traffic_mask, mask_token_expanded, x_traffic)

        # Concatenate back with time features (unmasked) if they exist
        if self.recon_dim < C:
            x_time = x[..., self.recon_dim:]  # [B, T, N, C - recon_dim] (tod, dow)
            x_masked = torch.cat([x_traffic_masked, x_time], dim=-1)
        else:
            x_masked = x_traffic_masked

        # Return mask for loss computation (only traffic part matters)
        full_mask = node_mask.expand(B, T, N, self.recon_dim)

        return x_masked, full_mask

    def forward(self, x: torch.Tensor, tod: torch.Tensor = None, dow: torch.Tensor = None) -> Dict:
        """
        Forward pass.

        Args:
            x: [B, T, N, C] input
               - 3-feat mode: [flow, tod, dow]
               - 5-feat mode: [flow, speed, occupancy, tod, dow]
            tod: [B, T] time of day indices (optional, can extract from x)
            dow: [B, T] day of week indices (optional, can extract from x)

        Returns:
            Dict with all components for loss computation
        """
        B, T, N, C = x.shape

        # Extract tod/dow using configured indices
        if tod is None and C > abs(self.tod_idx):
            tod = x[:, :, 0, self.tod_idx].long()  # [B, T] from first node
        if dow is None and C > abs(self.dow_idx):
            dow = x[:, :, 0, self.dow_idx].long()  # [B, T] from first node

        # Create two views with different masking for contrastive
        x_masked1, mask1 = self.apply_masking(x)
        x_masked2, mask2 = self.apply_masking(x)

        # Encode both views
        z_context1, z_self1 = self.encoder(x_masked1, return_disentangled=True)
        z_context2, z_self2 = self.encoder(x_masked2, return_disentangled=True)

        # Pool temporal dimension for contrastive loss
        # [B, T, N, D] -> [B, N, D]
        z_context1_pooled = z_context1.mean(dim=1)
        z_context2_pooled = z_context2.mean(dim=1)

        # Project for contrastive loss
        z1_proj = self.proj_head(z_context1_pooled)  # [B, N, D]
        z2_proj = self.proj_head(z_context2_pooled)  # [B, N, D]

        # Reconstruction from z_self (using first view)
        # Only reconstructs first recon_dim features (traffic features)
        x_recon = self.recon_head(z_self1)  # [B, T, N, recon_dim]

        # Extract only traffic features from original for loss computation
        x_traffic_original = x[..., :self.recon_dim]  # [B, T, N, recon_dim]

        return {
            'x_original': x_traffic_original,  # Only traffic features
            'x_recon': x_recon,
            'mask': mask1,
            'z1': z1_proj,  # For contrastive loss
            'z2': z2_proj,  # For contrastive loss
            'z_context': z_context1,
            'z_self': z_self1,
            'tod': tod,
            'dow': dow,
        }

    def compute_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        tod: torch.Tensor = None,
        dow: torch.Tensor = None,
    ) -> torch.Tensor:
        """Temporal-aware contrastive loss on z_context."""
        # [B, N, D] -> [B*N, D]
        B, N, D = z1.shape
        z1 = z1.reshape(B * N, D)
        z2 = z2.reshape(B * N, D)

        # Expand tod/dow for each node
        if tod is not None:
            if tod.dim() == 2:
                tod = tod[:, -1]  # Use last timestep
            tod = tod.unsqueeze(1).expand(B, N).reshape(B * N)
        if dow is not None:
            if dow.dim() == 2:
                dow = dow[:, -1]
            dow = dow.unsqueeze(1).expand(B, N).reshape(B * N)

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.shape[0]

        # Similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature

        # Compute temporal-aware negative weights
        weights = self._compute_negative_weights(tod, dow, batch_size, z1.device)

        # Mask diagonal
        mask = torch.eye(batch_size, device=z1.device).bool()
        weights = weights.masked_fill(mask, 0)

        # InfoNCE with weighted negatives
        pos_sim = torch.diag(sim_matrix)
        exp_sim = torch.exp(sim_matrix)
        weighted_neg_sum = (exp_sim * weights).sum(dim=1)

        loss = -pos_sim + torch.log(torch.exp(pos_sim) + weighted_neg_sum + 1e-8)
        return loss.mean()

    def _compute_negative_weights(
        self,
        tod: torch.Tensor,
        dow: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute negative weights based on temporal similarity."""
        weights = torch.ones(batch_size, batch_size, device=device)

        if tod is not None:
            tod_hours = (tod.float() / 12).long()
            tod_same = (tod_hours.unsqueeze(0) == tod_hours.unsqueeze(1)).float()
            tod_diff = 1 - tod_same
            tod_weight = tod_same * self.soft_positive_weight + tod_diff * 1.0
            weights = weights * (self.tod_weight * tod_weight + (1 - self.tod_weight))

        if dow is not None:
            dow_same = (dow.unsqueeze(0) == dow.unsqueeze(1)).float()
            dow_diff = 1 - dow_same
            is_weekday = (dow < 5).float()
            same_type = (is_weekday.unsqueeze(0) == is_weekday.unsqueeze(1)).float()
            dow_weight = dow_same * self.soft_positive_weight + dow_diff * same_type + dow_diff * (1 - same_type) * 1.2
            weights = weights * (self.dow_weight * dow_weight + (1 - self.dow_weight))

        return weights

    def compute_reconstruction_loss(
        self,
        x_original: torch.Tensor,
        x_recon: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction loss on z_self (masked positions only)."""
        return F.mse_loss(x_recon[mask], x_original[mask])

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode without masking (for downstream)."""
        return self.encoder.encode(x)

    def encode_disentangled(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode without masking, return disentangled representations."""
        return self.encoder.encode_disentangled(x)

    def get_encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()
