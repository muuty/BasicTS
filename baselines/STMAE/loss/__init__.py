"""STMAE Loss functions."""

from .stmae_loss import (
    STMAELoss,
    StructureLoss,
    FeatureLoss,
    stmae_pretrain_loss,
    stmae_finetune_loss,
)

# Alias for compatibility
stmae_loss = stmae_pretrain_loss

__all__ = [
    'STMAELoss',
    'StructureLoss',
    'FeatureLoss',
    'stmae_pretrain_loss',
    'stmae_finetune_loss',
    'stmae_loss',
]
