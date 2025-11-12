from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import json

import numpy as np

from rl.env.env import FlowFreeEnv
from rl.env.config import (
    BoardShape,
    EnvConfig,
    DEFAULT_OBSERVATION,
    MaskConfig,
    ObservationSpec,
    RewardPreset,
)


@dataclass
class TrajectoryStep:
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict | None = None
    observation: dict | None = None


@dataclass
class Trajectory:
    config: EnvConfig
    steps: list[TrajectoryStep]

    def to_json(self) -> dict:
        return {
            "config": self._config_to_dict(),
            "steps": [self._step_to_dict(step) for step in self.steps],
        }

    def _config_to_dict(self) -> dict:
        payload = asdict(self.config)
        payload["reward"]["components"] = list(payload["reward"]["components"])
        if "channels" in payload["observation"]:
            payload["observation"]["channels"] = list(payload["observation"]["channels"])
        return payload

    @staticmethod
    def _step_to_dict(step: TrajectoryStep) -> dict:
        return {
            "action": step.action,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "info": step.info,
            "observation": step.observation,
        }


def run_episode(env: FlowFreeEnv, policy: Callable[[dict], int]) -> Trajectory:
    obs, _ = env.reset()
    steps: list[TrajectoryStep] = []

    while True:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        steps.append(
            TrajectoryStep(
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=None,
                observation=None,
            )
        )
        obs = next_obs
        if terminated or truncated:
            break

    return Trajectory(config=env.config, steps=steps)


def save_trajectory(traj: Trajectory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(traj.to_json(), handle, indent=2, default=_json_default)


def load_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = _config_from_dict(payload["config"])
    steps: list[TrajectoryStep] = []
    for entry in payload["steps"]:
        steps.append(
            TrajectoryStep(
                action=int(entry["action"]),
                reward=float(entry["reward"]),
                terminated=bool(entry["terminated"]),
                truncated=bool(entry["truncated"]),
                info=None,
                observation=None,
            )
        )
    return Trajectory(config=config, steps=steps)


def _config_from_dict(data: dict) -> EnvConfig:
    if "shape" not in data:
        return _legacy_config_from_dict(data)

    shape = BoardShape(**data["shape"])
    observation = ObservationSpec(**data["observation"])
    mask = MaskConfig(**data["mask"])
    reward = RewardPreset(
        name=data["reward"]["name"],
        components=tuple(data["reward"]["components"]),
        params=dict(data["reward"]["params"]),
    )
    return EnvConfig(
        shape=shape,
        puzzle=data["puzzle"],
        reward=reward,
        observation=observation,
        mask=mask,
        max_steps=data.get("max_steps"),
        seed=data.get("seed"),
    )


def _legacy_config_from_dict(data: dict) -> EnvConfig:
    shape = BoardShape(
        width=data.get("width"),
        height=data.get("height", data.get("width")),
        color_count=data.get("color_count"),
    )
    reward = RewardPreset(
        name="legacy_trace",
        components=("potential", "completion", "constraints"),
        params={
            "move_penalty": -0.05,
            "distance_scale": 0.35,
            "complete_bonus": 1.0,
            "solve_bonus": 20.0,
            "invalid_penalty": -1.0,
            "dead_pocket_penalty": 0.0,
            "disconnect_penalty": -0.5,
            "degree_penalty": -0.3,
            "unsolved_penalty": -5.0,
            "undo_penalty": -0.1,
        },
    )
    return EnvConfig(
        shape=shape,
        puzzle=data.get("puzzle", ""),
        reward=reward,
        observation=DEFAULT_OBSERVATION,
        mask=MaskConfig(),
        max_steps=data.get("max_steps"),
        seed=data.get("seed"),
    )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


__all__ = ["Trajectory", "TrajectoryStep", "run_episode", "save_trajectory", "load_trajectory"]
