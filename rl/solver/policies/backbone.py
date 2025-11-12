from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FlowBackbone", "MLPBackbone"]


class ResidualBlock(nn.Module):
    """Simple residual conv block with optional dropout."""

    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out, inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.silu(out, inplace=True)


class FlowBackbone(nn.Module):
    """
    Lightweight residual CNN feature extractor used by policies and Q-networks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden: int = 128,
        blocks: int = 6,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, dropout=dropout) for _ in range(max(1, blocks))]
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(hidden, hidden // 4 if hidden >= 4 else hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden // 4 if hidden >= 4 else hidden, hidden, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        attn = self.attention(out)
        return out * attn


class MLPBackbone(nn.Module):
    """Stack of 1x1 convolutions acting as a per-cell MLP."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        channels = in_channels
        for _ in range(max(1, layers)):
            modules.append(nn.Conv2d(channels, hidden, kernel_size=1, bias=False))
            modules.append(nn.BatchNorm2d(hidden))
            modules.append(nn.SiLU(inplace=True))
            if dropout > 0.0:
                modules.append(nn.Dropout2d(p=dropout))
            channels = hidden
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
