"""
Decoders for STMAE.

InnerProductDecoder: Reconstructs adjacency matrix using inner product of node embeddings.
FeatureDecoder: Reconstructs input features from summary representation.
ForecastingDecoder: Produces future predictions for fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProductDecoder(nn.Module):
    """
    Decoder for structure reconstruction using inner product.

    Computes: adj = activation(z @ z.T)
    Used to reconstruct the adjacency matrix from node embeddings.
    """

    def __init__(
        self,
        dropout: float = 0.0,
        activation: str = 'none',
        with_proj: bool = False,
        hidden_dim: int = None,
    ):
        """
        Args:
            dropout: Dropout rate before inner product
            activation: Activation function ('sigmoid', 'none', or callable)
            with_proj: Whether to apply projection before inner product
            hidden_dim: Hidden dimension for projection (required if with_proj=True)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if activation == 'sigmoid':
            self.act = torch.sigmoid
        elif activation == 'none' or activation is None:
            self.act = lambda x: x
        else:
            self.act = activation

        if with_proj:
            assert hidden_dim is not None, "hidden_dim required when with_proj=True"
            self.proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct adjacency matrix from node embeddings.

        Args:
            z: Node embeddings [B, N, H]

        Returns:
            adj: Reconstructed adjacency [B, N, N]
        """
        z = self.dropout(z)
        # Project: [B, N, H] -> [B, H, N] -> Conv1d -> [B, H, N] -> [B, N, H]
        z = self.proj(z.permute(0, 2, 1)).permute(0, 2, 1)
        # Inner product: [B, N, H] @ [B, H, N] -> [B, N, N]
        adj = self.act(torch.bmm(z, z.permute(0, 2, 1)))
        return adj


class FeatureDecoder(nn.Module):
    """
    Decoder for feature reconstruction.

    Uses Conv2d to reconstruct input features from summary representation.
    Reconstructs [B, N, H] -> [B, T, N, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        seq_length: int,
    ):
        """
        Args:
            hidden_dim: Hidden dimension of encoder output
            output_dim: Output feature dimension (usually 1 for traffic data)
            seq_length: Sequence length to reconstruct
        """
        super().__init__()
        self.seq_length = seq_length
        self.output_dim = output_dim

        # Conv2d: input [B, 1, N, H] -> output [B, T*D, N, 1] -> reshape [B, T, N, D]
        self.decoder = nn.Conv2d(
            in_channels=1,
            out_channels=seq_length * output_dim,
            kernel_size=(1, hidden_dim),
            bias=True
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input features from summary.

        Args:
            summary: Encoder summary [B, N, H]

        Returns:
            reconstruction: Reconstructed features [B, T, N, D]
        """
        B, N, H = summary.shape
        # Reshape: [B, N, H] -> [B, 1, N, H]
        x = summary.unsqueeze(1)
        # Conv2d: [B, 1, N, H] -> [B, T*D, N, 1]
        x = self.decoder(x)
        # Reshape: [B, T*D, N, 1] -> [B, T, N, D]
        x = x.squeeze(-1)  # [B, T*D, N]
        x = x.view(B, self.seq_length, self.output_dim, N)  # [B, T, D, N]
        x = x.permute(0, 1, 3, 2)  # [B, T, N, D]
        return x


class ForecastingDecoder(nn.Module):
    """
    Decoder for forecasting task (fine-tuning).

    Produces future predictions from encoder output.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        horizon: int,
        use_mlp: bool = False,
    ):
        """
        Args:
            hidden_dim: Hidden dimension of encoder output
            output_dim: Output feature dimension
            horizon: Prediction horizon
            use_mlp: Whether to use MLP decoder (more capacity) or simple Conv2d
        """
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.use_mlp = use_mlp

        if not use_mlp:
            # Simple Conv2d decoder
            self.decoder = nn.Conv2d(
                in_channels=1,
                out_channels=horizon * output_dim,
                kernel_size=(1, hidden_dim),
                bias=True
            )
        else:
            # MLP decoder with more capacity
            self.decoder_1 = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 8,
                kernel_size=(1, 1),
                bias=True
            )
            self.decoder_2 = nn.Conv2d(
                in_channels=hidden_dim * 8,
                out_channels=horizon * output_dim,
                kernel_size=(1, 1),
                bias=True
            )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from encoder summary.

        Args:
            summary: Encoder summary [B, N, H] or [B, 1, N, H]

        Returns:
            prediction: Future predictions [B, T, N, D]
        """
        # Handle different input shapes
        if summary.dim() == 3:
            # [B, N, H] -> [B, 1, N, H]
            x = summary.unsqueeze(1)
        else:
            x = summary  # Already [B, 1, N, H]

        B = x.size(0)
        N = x.size(2)

        if not self.use_mlp:
            # Simple Conv2d: [B, 1, N, H] -> [B, T*D, N, 1]
            x = self.decoder(x)
        else:
            # MLP: [B, 1, N, H] -> [B, H, N, 1]
            x = x.permute(0, 3, 2, 1)  # [B, H, N, 1]
            x = F.relu(self.decoder_1(x))  # [B, H*8, N, 1]
            x = self.decoder_2(x)  # [B, T*D, N, 1]

        # Reshape: [B, T*D, N, 1] -> [B, T, N, D]
        x = x.squeeze(-1)  # [B, T*D, N]
        x = x.view(B, self.horizon, self.output_dim, N)  # [B, T, D, N]
        x = x.permute(0, 1, 3, 2)  # [B, T, N, D]

        return x
