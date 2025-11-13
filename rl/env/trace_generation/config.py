"""Configuration dataclass for trace generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl.env.config import RewardPreset


# Default reward preset for trace generation
TRACE_REWARD = RewardPreset(
    name="trace_generation",
    components=("potential", "completion", "constraints"),
    params={
        "move_penalty": -0.08,
        "distance_scale": 0.35,
        "complete_bonus": 3.0,
        "solve_bonus": 20.0,
        "invalid_penalty": -1.0,
        "dead_pocket_penalty": 0.0,
        "disconnect_penalty": -0.7500712307117836,
        "degree_penalty": -0.24601623361070557,
        "unsolved_penalty": -25.0,
        "undo_penalty": -0.1,
    },
)


@dataclass
class TraceGenConfig:
    """Configuration for trace generation from completed puzzles.

    Attributes:
        out_dir: Directory to store generated trajectories
        solver_name: Prefix used for saved trajectory files
        max_size: Maximum board size to process
        max_colors: Optional cap on color count
        force_overwrite: Whether to overwrite existing trajectory files
        verbose: Whether to log per-puzzle progress
        completion_mode: Strategy for ordering color completions
        variants: Number of color-order variants to export per puzzle (normal mode only)
        shuffle_colors: Whether to randomize color order for variants (normal mode only)
        reward_preset: Reward configuration for environment during replay
    """

    out_dir: Path = Path("data/rl_traces")
    solver_name: str = "rl_traces"
    max_size: int = 8
    max_colors: int | None = None
    force_overwrite: bool = False
    verbose: bool = False
    completion_mode: str = "normal"
    variants: int = 1
    shuffle_colors: bool = False
    reward_preset: RewardPreset = TRACE_REWARD


__all__ = ["TraceGenConfig", "TRACE_REWARD"]
