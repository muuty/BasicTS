# #!/usr/bin/env python3
# """STAEformer 서버 모델들"""

# import torch
# import torch.nn as nn

# import sys
# import os
# from ...base_models import BaseServerModel
# from ...layers import SelfAttentionLayer


# #!/usr/bin/env python3
# """STAEformer Hierarchical Server Model"""

# import torch
# import torch.nn as nn
# from ...base_models import BaseServerModel
# from ...layers import SelfAttentionLayer

# class STAEformerHierServerModel(BaseServerModel):
#     """
#     STAEformer-Hier 서버 모델
#     - 모든 클라이언트의 압축된 토큰(Pooled Tokens)을 받아 Global Spatial Attention 수행
#     - 입력: (B, T, Total_Tokens, D)
#     - 출력: (B, T, Total_Tokens, D)
#     """
    
#     def __init__(
#         self, 
#         num_nodes: int,  # 여기서는 Total Tokens 수를 의미 (dummy)
#         model_dim: int = 96,
#         feed_forward_dim: int = 256, 
#         num_heads: int = 4, 
#         num_layers: int = 1,
#         dropout: float = 0.1, 
#     ):
#         super().__init__(num_nodes, model_dim)
        
#         self.model_dim = model_dim
        
#         # Global Spatial Attention layers
#         # dim=2 (Token Dimension)을 따라 어텐션 수행
#         self.attn_layers_combined = nn.ModuleList([
#             SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
    
#     def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pooled_features: (B, T, Total_K, D)
#         Returns:
#             server_features: (B, T, Total_K, D)
#         """
#         x = pooled_features 
        
#         # Global Attention: 클라이언트 간의 정보 교환 발생
#         for attn in self.attn_layers_combined:
#             x = attn(x, dim=2)
        
#         return x
