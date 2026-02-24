"""
Cross-Variable Consistency Pre-training Model.

Pre-trains an encoder by masking one physical variable (flow, occupancy, or speed)
and reconstructing it from the remaining variables. Forces the encoder to learn
cross-variable physical relationships, producing noise-robust representations.

Key idea: traffic variables are physically related (fundamental diagram).
If one sensor channel fails, the others can compensate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_aware_encoder import ContextAwareEncoder


class CrossVariablePretrainModel(nn.Module):
    """
    Cross-Variable Consistency Pre-training.

    Architecture:
    - ContextAwareEncoder: temporal encoder (processes each node independently)
    - Reconstruction head: predicts masked physical variable from encoder output

    Training:
    1. Randomly mask 1 of 3 physical variables (flow/occ/speed) per sample
    2. Encode the masked input (tod/dow remain as temporal context)
    3. Reconstruct the masked variable
    4. Loss: MAE on masked positions only
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int = 5,
        output_dim: int = 1,
        d_model: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        num_physical_vars: int = 3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_physical_vars = num_physical_vars

        # Temporal encoder (reuse existing)
        self.encoder = ContextAwareEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )

        # Reconstruction head: predict physical variables from representation
        self.recon_head = nn.Linear(d_model, num_physical_vars)

        # Dummy predictor for basicts framework compatibility
        self.dummy_predictor = nn.Linear(d_model, output_len * output_dim)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> dict:
        """
        Args:
            history_data: [B, T, N, C] where C=5 (flow, occ, speed, tod, dow)
            future_data: [B, T', N, C] (unused, for framework compatibility)

        Returns:
            dict with 'prediction', 'recon_pred', 'recon_target', 'recon_mask'
        """
        B, T, N, C = history_data.shape

        if train:
            # Randomly mask 1 of num_physical_vars per sample (vectorized)
            mask_idx = torch.randint(0, self.num_physical_vars, (B,), device=history_data.device)
            mask = F.one_hot(mask_idx, self.num_physical_vars).float().view(B, 1, 1, self.num_physical_vars)

            x_masked = history_data.clone()
            x_masked[..., :self.num_physical_vars] = x_masked[..., :self.num_physical_vars] * (1 - mask)
        else:
            x_masked = history_data
            mask = torch.ones(B, 1, 1, self.num_physical_vars, device=history_data.device)

        # Encode
        z = self.encoder(x_masked)  # [B, T, N, d_model]

        # Reconstruct physical variables
        recon = self.recon_head(z)  # [B, T, N, num_physical_vars]

        # Dummy prediction for framework compatibility
        dummy = self.dummy_predictor(z[:, -1, :, :])  # [B, N, output_len * output_dim]
        dummy = dummy.reshape(B, N, self.output_len, self.output_dim)
        dummy = dummy.permute(0, 2, 1, 3)  # [B, T', N, C]

        return {
            'prediction': dummy,
            'recon_pred': recon,
            'recon_target': history_data[..., :self.num_physical_vars],
            'recon_mask': mask,
        }
