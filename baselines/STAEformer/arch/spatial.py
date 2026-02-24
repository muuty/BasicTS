# spatial.py
import torch
import torch.nn as nn
from typing import Optional

from baselines.layers import SelfAttentionLayer


class STAEformerSpatial(nn.Module):
    """
    STAEformer Spatial: Spatial Attention
    
    원본 STAEformer의 중간 부분:
    - Spatial Attention (attn_layers_s)
    """
    
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Spatial Attention layers
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        graph: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, model_dim)
            graph: Not used in STAEformer (for interface compatibility)
            
        Returns:
            spatial_features: (B, T, N, model_dim)
        """
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        
        return x