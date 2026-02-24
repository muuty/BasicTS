"""
Unified Pretrain Model.

Config:
    model_config = {
        'encoder': {...},
        'adj_path': 'datasets/.../adj_mx.pkl',  # optional
        'masking': {
            'spatial': {'enabled': True, 'ratio': 0.15},
            'feature': {'enabled': True, 'ratio': 0.30},
        },
        'heads': {'contrastive': True, 'reconstruction': True},
    }
"""
import pickle
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple

from .augmentations import NodeMasking, FeatureMasking, TemporalMasking, NeighborhoodMasking
from .base_encoder import build_encoder


class UnifiedPretrainModel(nn.Module):
    """Unified pretraining model with configurable masking and heads."""

    def __init__(self, **kwargs):
        super().__init__()
        # Accept kwargs from basicts which unpacks MODEL.PARAM
        config = kwargs
        self.config = config

        # Load adjacency matrix if specified
        adj_path = config.get('adj_path')
        if adj_path:
            with open(adj_path, 'rb') as f:
                data = pickle.load(f)
            # Standard format: (sensor_ids, sensor_id_to_ind, adj_mx) or just array
            adj_mx = data[2] if hasattr(data, '__getitem__') and not hasattr(data, 'shape') else data
            # Handle sparse matrices (scipy) and dense arrays
            adj_mx = np.asarray(adj_mx.toarray() if sp.issparse(adj_mx) else adj_mx)
            self.register_buffer('adj', torch.from_numpy(adj_mx).float())
        else:
            self.adj = None

        # Build encoder
        encoder_cfg = config['encoder']
        self.encoder = build_encoder(encoder_cfg)
        self.d_model = self.encoder.d_model
        self.input_dim = encoder_cfg.get('input_dim', 3)

        # Learnable mask token (MAE-style)
        # Shape: [1, 1, 1, input_dim] - will be broadcast to [B, T, N, C]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Build masking modules (with return_mask=True for MAE-style)
        self.masking_modules = nn.ModuleDict()
        masking_cfg = config.get('masking', {})
        if masking_cfg.get('spatial', {}).get('enabled'):
            self.masking_modules['spatial'] = NodeMasking(mask_ratio=masking_cfg['spatial']['ratio'], return_mask=True)
        if masking_cfg.get('feature', {}).get('enabled'):
            self.masking_modules['feature'] = FeatureMasking(mask_ratio=masking_cfg['feature']['ratio'], return_mask=True)
        if masking_cfg.get('temporal', {}).get('enabled'):
            self.masking_modules['temporal'] = TemporalMasking(mask_ratio=masking_cfg['temporal']['ratio'], return_mask=True)
        if masking_cfg.get('neighborhood', {}).get('enabled'):
            cfg = masking_cfg['neighborhood']
            self.masking_modules['neighborhood'] = NeighborhoodMasking(
                adj_mx_path=cfg.get('adj_path'), mask_ratio=cfg['ratio'], neighbors_per_seed=cfg.get('neighbors', 10))

        # Heads
        heads_cfg = config.get('heads', {})
        proj_dim = config.get('proj_dim', self.d_model)

        self.proj_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, proj_dim)
        ) if heads_cfg.get('contrastive', True) else None

        self.recon_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.input_dim)
        ) if heads_cfg.get('reconstruction', False) else None

    def apply_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply configured masking with learnable mask token (MAE-style).

        Returns (x_masked, mask) where:
            - x_masked: input with masked positions replaced by learnable mask_token
            - mask: boolean tensor, True = masked position (to be reconstructed)
        """
        B, T, N, C = x.shape

        if not self.masking_modules:
            return x, torch.zeros(B, T, N, C, dtype=torch.bool, device=x.device)

        # Collect masks from all masking modules
        combined_mask = torch.zeros(B, T, N, C, dtype=torch.bool, device=x.device)

        for name, module in self.masking_modules.items():
            if name == 'neighborhood':
                # NeighborhoodMasking doesn't support return_mask yet, use legacy
                x_temp = module(x)
                module_mask = (x_temp == 0) & (x != 0)
            else:
                module.train()
                _, module_mask = module(x)
            combined_mask = combined_mask | module_mask

        # Apply learnable mask token to masked positions (MAE-style)
        mask_token_expanded = self.mask_token.expand(B, T, N, C)
        x_masked = torch.where(combined_mask, mask_token_expanded, x)

        return x_masked, combined_mask

    def forward(self, x: torch.Tensor, return_reconstruction: bool = True) -> Dict:
        """Forward pass. Returns dict with z1, z2, x_recon, x_original, mask."""
        x1, mask1 = self.apply_masking(x)
        x2, mask2 = self.apply_masking(x)
        # Pass adj as keyword argument (some encoders use it, others don't)
        h1 = self.encoder.encode(x1, adj=self.adj)
        h2 = self.encoder.encode(x2, adj=self.adj)

        result = {'x_original': x, 'mask1': mask1, 'mask2': mask2}

        h1_pooled = h1.mean(dim=1) if h1.dim() == 4 else h1
        h2_pooled = h2.mean(dim=1) if h2.dim() == 4 else h2
        result['h1'], result['h2'] = h1_pooled, h2_pooled

        if self.proj_head:
            result['z1'] = self.proj_head(h1_pooled)
            result['z2'] = self.proj_head(h2_pooled)

        if self.recon_head and return_reconstruction:
            h_for_recon = h1.unsqueeze(1).expand(-1, x.shape[1], -1, -1) if h1.dim() == 3 else h1
            result['x_recon'] = self.recon_head(h_for_recon)
            result['mask'] = mask1

        return result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without masking (for downstream use)."""
        return self.encoder.encode(x, adj=self.adj)

    def get_encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()

    def get_config(self) -> dict:
        return self.config
