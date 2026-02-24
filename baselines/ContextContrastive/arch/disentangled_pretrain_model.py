"""
Disentangled Pretrain Model.

Pretrains disentangled representations (z_context, z_self) using reconstruction loss.
Optionally enforces orthogonality between the two representations.

Config:
    model_config = {
        'input_dim': 3,
        'd_model': 64,
        'num_layers': 2,
        'nhead': 4,
        'dropout': 0.1,
        'fusion': 'concat',  # 'concat', 'sum', 'gate'
        'use_orthogonality': False,  # v1: False, v2: True
        'ortho_weight': 0.1,
        'mask_ratio': 0.5,
    }
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .disentangled_encoder import DisentangledEncoder


class DisentangledPretrainModel(nn.Module):
    """Disentangled pretraining model with reconstruction loss."""

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        fusion: str = 'concat',
        use_orthogonality: bool = False,
        ortho_weight: float = 0.1,
        mask_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.use_orthogonality = use_orthogonality

        # Disentangled encoder
        self.encoder = DisentangledEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            fusion=fusion,
            use_orthogonality=use_orthogonality,
            ortho_weight=ortho_weight,
        )

        # Learnable mask token (MAE-style)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Reconstruction heads (one for each representation path)
        output_dim = self.encoder.output_dim
        self.recon_head = nn.Sequential(
            nn.Linear(output_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )

        # Optional: separate reconstruction heads for context and self
        self.recon_head_context = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )
        self.recon_head_self = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )

    def apply_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial masking with learnable mask token.

        Returns:
            x_masked: Input with masked positions replaced by mask token
            mask: Boolean tensor, True = masked position
        """
        B, T, N, C = x.shape

        # Create spatial mask (mask entire nodes across all time steps)
        mask = torch.rand(B, 1, N, 1, device=x.device) < self.mask_ratio
        mask = mask.expand(B, T, N, C)

        # Apply learnable mask token
        mask_token_expanded = self.mask_token.expand(B, T, N, C)
        x_masked = torch.where(mask, mask_token_expanded, x)

        return x_masked, mask

    def forward(self, x: torch.Tensor, return_all: bool = True) -> Dict:
        """Forward pass.

        Returns dict with:
            - x_original: Original input
            - x_recon: Reconstructed input (from fused representation)
            - x_recon_context: Reconstructed from context only
            - x_recon_self: Reconstructed from self only
            - z_context, z_self: Disentangled representations
            - mask: Boolean mask
            - ortho_loss: Orthogonality loss (if enabled)
        """
        # Apply masking
        x_masked, mask = self.apply_masking(x)

        # Encode (get both representations)
        z_context, z_self = self.encoder(x_masked, return_disentangled=True)

        result = {
            'x_original': x,
            'mask': mask,
            'z_context': z_context,
            'z_self': z_self,
        }

        # Fused reconstruction
        if self.encoder.fusion == 'concat':
            z_fused = torch.cat([z_context, z_self], dim=-1)
        elif self.encoder.fusion == 'gate':
            gate = self.encoder.gate(torch.cat([z_context, z_self], dim=-1))
            z_fused = gate * z_context + (1 - gate) * z_self
        else:
            z_fused = z_context + z_self

        result['x_recon'] = self.recon_head(z_fused)

        if return_all:
            # Individual reconstructions (for analysis)
            result['x_recon_context'] = self.recon_head_context(z_context)
            result['x_recon_self'] = self.recon_head_self(z_self)

        # Orthogonality loss
        if self.use_orthogonality:
            result['ortho_loss'] = self.encoder.compute_orthogonality_loss(z_context, z_self)
        else:
            # Use a zero tensor that doesn't break gradient computation
            result['ortho_loss'] = torch.zeros(1, device=x.device, requires_grad=False)

        return result

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode without masking (for downstream use). Returns fused representation."""
        return self.encoder.encode(x)

    def encode_disentangled(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode without masking, returning disentangled representations."""
        return self.encoder.encode_disentangled(x)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder state dict for transfer learning."""
        return self.encoder.state_dict()

    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'mask_ratio': self.mask_ratio,
            'use_orthogonality': self.use_orthogonality,
        }
