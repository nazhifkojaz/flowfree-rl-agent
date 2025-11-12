from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.solver.core import ACTION_DIM, MAX_CHANNELS

from .backbone import FlowBackbone

__all__ = ["FlowPolicy", "masked_cross_entropy", "save_policy", "load_policy"]


class FlowPolicy(nn.Module):
    """Convolutional policy used for supervised warm-start and DQN initialization."""

    def __init__(
        self,
        in_channels: int = MAX_CHANNELS,
        hidden: int = 128,
        blocks: int = 6,
        dropout: float = 0.1,
        action_dim: int | None = None,
        use_attention: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim or ACTION_DIM
        self.backbone = FlowBackbone(
            in_channels=in_channels, hidden=hidden, blocks=blocks, dropout=dropout, use_attention=use_attention
        )
        head_dim = hidden * 2
        self.policy_head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.action_dim),
        )

    @staticmethod
    def _global_features(features: torch.Tensor) -> torch.Tensor:
        avg = torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        maxv = torch.flatten(F.adaptive_max_pool2d(features, 1), 1)
        return torch.cat([avg, maxv], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        global_feat = self._global_features(features)
        logits = self.policy_head(global_feat)
        return logits.masked_fill(~mask.bool(), -1e9)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(F.log_softmax(logits, dim=-1), targets)


def save_policy(model: FlowPolicy, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_policy(model: FlowPolicy, path: Path, *, map_location: str | torch.device | None = None) -> None:
    state = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(state)
