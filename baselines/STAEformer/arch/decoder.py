# decoder.py
import torch
import torch.nn as nn


class STAEformerDecoder(nn.Module):
    """
    STAEformer Decoder: Output Projection
    
    원본 STAEformer의 뒷부분:
    - Mixed projection 또는 temporal + output projection
    """
    
    def __init__(
        self,
        num_nodes: int,
        model_dim: int,
        in_steps: int = 12,
        out_steps: int = 12,
        output_dim: int = 1,
        use_mixed_proj: bool = True,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.use_mixed_proj = use_mixed_proj
        
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * model_dim, 
                out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(model_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, model_dim)
            
        Returns:
            prediction: (B, out_steps, N, output_dim)
        """
        batch_size = x.shape[0]
        num_nodes = x.shape[2]
        
        if self.use_mixed_proj:
            # (B, T, N, D) -> (B, N, T, D)
            out = x.transpose(1, 2)
            # (B, N, T, D) -> (B, N, T*D)
            out = out.reshape(batch_size, num_nodes, self.in_steps * self.model_dim)
            # (B, N, T*D) -> (B, N, out_steps * output_dim)
            out = self.output_proj(out)
            # (B, N, out_steps * output_dim) -> (B, N, out_steps, output_dim)
            out = out.view(batch_size, num_nodes, self.out_steps, self.output_dim)
            # (B, N, out_steps, output_dim) -> (B, out_steps, N, output_dim)
            out = out.transpose(1, 2)
        else:
            # (B, T, N, D) -> (B, D, N, T)
            out = x.transpose(1, 3)
            # (B, D, N, T) -> (B, D, N, out_steps)
            out = self.temporal_proj(out)
            # (B, D, N, out_steps) -> (B, out_steps, N, D) -> (B, out_steps, N, output_dim)
            out = self.output_proj(out.transpose(1, 3))
        
        return out