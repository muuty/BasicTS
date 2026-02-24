"""
Dilated Convolutional Encoder for TS2Vec

Adapted from the original TS2Vec implementation to handle
spatial-temporal data with shape [B, T, N, C].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv(nn.Module):
    """1D convolution with same padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlock(nn.Module):
    """Residual convolutional block with dilated convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, final: bool = False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    """
    Dilated Convolutional Encoder.

    Uses exponentially increasing dilation rates to capture
    multi-scale temporal patterns.
    """

    def __init__(self, in_channels: int, channels: list, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
