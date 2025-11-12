from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.solver.constants import ACTIONS_PER_COLOR, MAX_COLORS
from .backbone import FlowBackbone

__all__ = ["FlowQNetwork"]


class FlowQNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        blocks: int = 6,
        dropout: float = 0.1,
        attn_layers: int = 2,
        attn_heads: int = 4,
        use_dueling: bool = False,
    ):
        super().__init__()
        self.expected_channels = in_channels
        self.max_colors = MAX_COLORS
        self.use_dueling = use_dueling

        self.backbone = FlowBackbone(in_channels=in_channels, hidden=hidden, blocks=blocks, dropout=dropout)
        self.global_dim = hidden * 2
        self.hidden = hidden
        self.attn_dim = hidden

        feature_dim = self.global_dim + hidden * 5
        self.color_embedding = nn.Embedding(self.max_colors, hidden)
        self.action_embedding = nn.Embedding(ACTIONS_PER_COLOR, hidden)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.SiLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=attn_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, 1),
            )

        self.adv_head = make_head()
        self.value_head = make_head() if use_dueling else None

    @staticmethod
    def _global_features(features: torch.Tensor) -> torch.Tensor:
        avg = torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        maxv = torch.flatten(F.adaptive_max_pool2d(features, 1), 1)
        return torch.cat([avg, maxv], dim=1)

    @staticmethod
    def _shift(mask: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
        shifted = torch.zeros_like(mask)
        h, w = mask.shape[-2:]

        src_h_start = max(0, -dr)
        src_h_end = h - max(0, dr)
        dst_h_start = max(0, dr)
        dst_h_end = h - max(0, -dr)

        src_w_start = max(0, -dc)
        src_w_end = w - max(0, dc)
        dst_w_start = max(0, dc)
        dst_w_end = w - max(0, -dc)

        shifted[:, :, dst_h_start:dst_h_end, dst_w_start:dst_w_end] = mask[:, :, src_h_start:src_h_end, src_w_start:src_w_end]
        return shifted

    def forward(
        self,
        x: torch.Tensor,
        *,
        head_masks: torch.Tensor,
        target_masks: torch.Tensor,
        color_counts: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)
        batch_size, hidden_dim, height, width = features.shape

        head_masks = head_masks[:, : self.max_colors, :height, :width]
        target_masks = target_masks[:, : self.max_colors, :height, :width]

        # Aggregated features
        def aggregate(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_sum = mask.flatten(2).sum(-1, keepdim=True)
            weighted = (feat.unsqueeze(1) * mask.unsqueeze(2)).flatten(3).sum(-1)
            return weighted / (mask_sum + 1e-6)

        head_feat = aggregate(features, head_masks)
        target_feat = aggregate(features, target_masks)

        directional_masks = [
            self._shift(head_masks, -1, 0),
            self._shift(head_masks, 0, 1),
            self._shift(head_masks, 1, 0),
            self._shift(head_masks, 0, -1),
        ]
        directional_feats = torch.stack(
            [aggregate(features, mask) for mask in directional_masks],
            dim=2,
        )  # (B, C, 4, hidden)
        undo_feat = head_feat.unsqueeze(2)  # (B, C, 1, hidden)
        action_feats = torch.cat([directional_feats, undo_feat], dim=2)  # (B, C, 5, hidden)

        global_feat = self._global_features(features)
        global_expanded = global_feat.unsqueeze(1).unsqueeze(2).expand(-1, self.max_colors, ACTIONS_PER_COLOR, -1)
        head_expanded = head_feat.unsqueeze(2).expand(-1, self.max_colors, ACTIONS_PER_COLOR, -1)
        target_expanded = target_feat.unsqueeze(2).expand(-1, self.max_colors, ACTIONS_PER_COLOR, -1)

        color_indices = torch.arange(self.max_colors, device=x.device)
        color_embeddings = (
            self.color_embedding(color_indices)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, ACTIONS_PER_COLOR, -1)
        )
        action_indices = torch.arange(ACTIONS_PER_COLOR, device=x.device)
        action_embeddings = (
            self.action_embedding(action_indices)
            .reshape(1, 1, ACTIONS_PER_COLOR, -1)
            .expand(batch_size, self.max_colors, -1, -1)
        )

        features_concat = torch.cat(
            [
                global_expanded,
                head_expanded,
                target_expanded,
                action_feats,
                color_embeddings,
                action_embeddings,
            ],
            dim=3,
        )

        flat = features_concat.view(batch_size, self.max_colors * ACTIONS_PER_COLOR, -1)
        tokens = self.token_proj(flat)

        padding_mask = torch.arange(self.max_colors, device=x.device).unsqueeze(0) >= color_counts.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, ACTIONS_PER_COLOR).reshape(
            batch_size, self.max_colors * ACTIONS_PER_COLOR
        )

        attn_out = self.transformer(tokens, src_key_padding_mask=padding_mask)
        advantage = self.adv_head(attn_out).squeeze(-1)

        color_mask = torch.arange(self.max_colors, device=x.device).unsqueeze(0) < color_counts.unsqueeze(1)
        color_mask = color_mask.unsqueeze(-1).expand(-1, -1, ACTIONS_PER_COLOR).reshape(
            batch_size, self.max_colors * ACTIONS_PER_COLOR
        )

        if self.use_dueling and self.value_head is not None:
            mask_float = color_mask.float()
            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            masked_attn = attn_out * mask_float.unsqueeze(-1)
            state_feat = masked_attn.sum(dim=1) / denom
            state_value = self.value_head(state_feat)
            adv_mean = (advantage * mask_float).sum(dim=1, keepdim=True) / denom
            q_values = state_value + advantage - adv_mean
        else:
            q_values = advantage

        q_values = q_values.masked_fill(~color_mask, 0.0)
        return q_values
