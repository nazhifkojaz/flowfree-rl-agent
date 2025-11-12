from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from rl.env.config import default_max_steps

from .constants import (
    ACTION_DIM,
    ACTIONS_PER_COLOR,
    BASE_OBS_CHANNELS,
    EXTRA_OBS_CHANNELS,
    MAX_CHANNELS,
    MAX_COLORS,
    MAX_SIZE,
)


@dataclass
class EncodedObservation:
    state: torch.Tensor
    head_mask: torch.Tensor
    target_mask: torch.Tensor
    color_count: int
    height: int
    width: int


def get_action_dim(color_count: int) -> int:
    return color_count * ACTIONS_PER_COLOR


def get_channels_for_size(size: int) -> int:
    return MAX_CHANNELS


def _coordinate_planes(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    row_coords = np.linspace(-1.0, 1.0, num=height, dtype=np.float32).reshape(1, height, 1)
    row_plane = np.broadcast_to(row_coords, (1, height, width))
    col_coords = np.linspace(-1.0, 1.0, num=width, dtype=np.float32).reshape(1, 1, width)
    col_plane = np.broadcast_to(col_coords, (1, height, width))
    return row_plane, col_plane


def encode_observation(
    observation: dict,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> EncodedObservation:
    base_tensor = observation["tensor"].astype(np.float32)
    channels, height, width = base_tensor.shape

    color_count = int(observation.get("color_count", max(1, channels // 4)))
    max_channels = get_channels_for_size(max(height, width))
    padded = np.zeros((max_channels, MAX_SIZE, MAX_SIZE), dtype=np.float32)
    padded[:channels, :height, :width] = base_tensor

    row_plane, col_plane = _coordinate_planes(height, width)
    extra_idx = BASE_OBS_CHANNELS
    if extra_idx < max_channels:
        padded[extra_idx, :height, :width] = row_plane
        extra_idx += 1
        if extra_idx < max_channels:
            padded[extra_idx, :height, :width] = col_plane
            extra_idx += 1

    if max_channels > BASE_OBS_CHANNELS + 2:
        steps = float(observation.get("steps", 0))
        inferred_max_steps = float(
            observation.get("max_steps") or default_max_steps(width, height)
        )
        if inferred_max_steps <= 0.0:
            inferred_max_steps = float(default_max_steps(width, height))
        step_fraction = steps / inferred_max_steps

        board_fraction = float(height) / float(MAX_SIZE)
        color_fraction = color_count / float(MAX_COLORS)

        free_mask = base_tensor[-2] if channels >= 2 else np.zeros((height, width), dtype=np.float32)
        empty_fraction = float(free_mask.mean()) if free_mask.size else 0.0

        remaining = np.asarray(observation.get("remaining_lengths", []), dtype=np.float32)
        mean_remaining = float(remaining.mean() / MAX_SIZE) if remaining.size else 0.0

        for value in (step_fraction, board_fraction, color_fraction, empty_fraction, mean_remaining):
            if extra_idx < max_channels:
                padded[extra_idx, :height, :width] = value
                extra_idx += 1

        remaining_norm = np.zeros(MAX_COLORS, dtype=np.float32)
        count = min(len(remaining), MAX_COLORS)
        if count:
            remaining_norm[:count] = remaining[:count] / MAX_SIZE
        if extra_idx + MAX_COLORS <= max_channels:
            padded[extra_idx : extra_idx + MAX_COLORS, :height, :width] = remaining_norm.reshape(
                MAX_COLORS, 1, 1
            )
            extra_idx += MAX_COLORS

    state = torch.from_numpy(padded).to(device=device, dtype=dtype)

    head_mask_np = np.zeros((MAX_COLORS, MAX_SIZE, MAX_SIZE), dtype=np.float32)
    target_mask_np = np.zeros((MAX_COLORS, MAX_SIZE, MAX_SIZE), dtype=np.float32)

    head_start = color_count * 2  # occupancy + endpoints
    head_end = head_start + color_count * 2
    if head_end <= base_tensor.shape[0]:
        head_channels = base_tensor[head_start:head_end]
        active_heads = head_channels[0::2]
        target_heads = head_channels[1::2]
        head_mask_np[:color_count, :height, :width] = active_heads
        target_mask_np[:color_count, :height, :width] = target_heads

    head_mask = torch.from_numpy(head_mask_np).to(device=device, dtype=dtype)
    target_mask = torch.from_numpy(target_mask_np).to(device=device, dtype=dtype)

    return EncodedObservation(
        state=state,
        head_mask=head_mask,
        target_mask=target_mask,
        color_count=color_count,
        height=height,
        width=width,
    )


def mask_to_tensor(mask: np.ndarray | Iterable[int], *, device: torch.device, color_count: int | None = None) -> torch.Tensor:
    as_array = np.asarray(mask, dtype=np.float32)
    action_dim = ACTION_DIM
    padded = np.zeros(action_dim, dtype=np.float32)
    padded[: len(as_array)] = as_array
    return torch.from_numpy(padded).to(device)


def obs_to_tensor(
    observation: dict,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return encode_observation(observation, device=device, dtype=dtype).state


__all__ = [
    "encode_observation",
    "obs_to_tensor",
    "mask_to_tensor",
    "EncodedObservation",
    "get_action_dim",
]
