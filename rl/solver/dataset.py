from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rl.env.env import FlowFreeEnv
from rl.env.config import BoardShape, EnvConfig, DEFAULT_OBSERVATION, MaskConfig, POTENTIAL_REWARD
from rl.env.trace import load_trajectory

from .constants import ACTION_DIM, MAX_COLORS, MAX_SIZE
from .observation import encode_observation, mask_to_tensor


@dataclass
class TrajectoryExample:
    state: torch.Tensor  # (C, H, W)
    head_mask: torch.Tensor  # (MAX_COLORS, MAX_SIZE, MAX_SIZE)
    target_mask: torch.Tensor  # same as head_mask
    remaining: torch.Tensor  # (MAX_COLORS,)
    steps: torch.Tensor  # scalar tensor
    max_steps: torch.Tensor  # scalar tensor
    colour_count: torch.Tensor  # scalar tensor
    action_mask: torch.Tensor  # (ACTION_DIM,)
    action: int


class SupervisedTrajectoryDataset(Dataset[TrajectoryExample]):
    """Dataset of (state, mask, action) pairs built from saved trajectories."""

    def __init__(self, trace_paths: Sequence[Path], dtype: torch.dtype = torch.float32):
        self._dtype = dtype
        self._examples: list[TrajectoryExample] = []
        for path in trace_paths:
            self._load_trace(Path(path))

    def _load_trace(self, path: Path) -> None:
        trajectory = load_trajectory(path)
        env_config = _ensure_env_config(trajectory.config)
        env = FlowFreeEnv(env_config)
        obs, _ = env.reset()

        for step in trajectory.steps:
            encoded = encode_observation(obs, device=torch.device("cpu"), dtype=self._dtype)
            tensor_t = encoded.state
            head_mask = encoded.head_mask
            target_mask = encoded.target_mask

            remaining = torch.zeros(MAX_COLORS, dtype=self._dtype)
            rem = obs.get("remaining_lengths", [])
            if rem is not None:
                rem_arr = np.asarray(rem, dtype=np.float32)
                if rem_arr.size:
                    rem_t = torch.from_numpy(rem_arr)
                    count = min(rem_t.shape[0], MAX_COLORS)
                    remaining[:count] = rem_t[:count]

            steps = torch.tensor(float(obs.get("steps", 0.0)), dtype=self._dtype)
            inferred_default = encoded.height * encoded.width + 12
            max_steps_value = float(obs.get("max_steps") or inferred_default)
            max_steps = torch.tensor(max_steps_value, dtype=self._dtype)
            colour_count = torch.tensor(float(obs.get("color_count", encoded.color_count)), dtype=self._dtype)

            action_mask = mask_to_tensor(
                obs["action_mask"],
                device=torch.device("cpu"),
                color_count=obs.get("color_count"),
            )
            action = int(step.action)

            self._examples.append(
                TrajectoryExample(
                    state=tensor_t,
                    head_mask=head_mask,
                    target_mask=target_mask,
                    remaining=remaining,
                    steps=steps,
                    max_steps=max_steps,
                    colour_count=colour_count,
                    action_mask=action_mask,
                    action=action,
                )
            )

            obs, _, _, _, _ = env.step(step.action)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrajectoryExample:
        return self._examples[idx]


def _ensure_env_config(config: EnvConfig | object) -> EnvConfig:
    if isinstance(config, EnvConfig):
        return config

    width = getattr(config, "width")
    height = getattr(config, "height")
    color_count = getattr(config, "color_count")
    max_steps = getattr(config, "max_steps", None)
    puzzle = getattr(config, "puzzle")
    shape = BoardShape(width=width, height=height, color_count=color_count)
    return EnvConfig(
        shape=shape,
        puzzle=puzzle,
        reward=POTENTIAL_REWARD,
        observation=DEFAULT_OBSERVATION,
        mask=MaskConfig(),
        max_steps=max_steps,
    )


def discover_trace_files(root: Path, *, exts: Iterable[str] | None = None) -> list[Path]:
    extensions = {*(exts or [".json"])}
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in extensions:
            out.append(path)
    return out
