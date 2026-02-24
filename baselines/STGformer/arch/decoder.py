import torch
import torch.nn as nn
from typing import Tuple


class STGformerDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        out_steps: int,
        output_dim: int,
        in_steps: int,
        kernel_size: int,
        mlp_ratio: float,
        dropout: float,
        num_layers: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.in_steps = in_steps
        self.kernel_size = kernel_size

        effective_steps = in_steps - sum(k - 1 for k in kernel_size)
        self.encoder_proj = nn.Linear(
            effective_steps * model_dim,
            model_dim,
        )
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, int(model_dim * mlp_ratio)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(model_dim * mlp_ratio), model_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(model_dim, out_steps * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)
        out = self.output_proj(x).view(
            batch_size, -1, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)
        return out


































