"""Rollout recording configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RolloutConfig:
    """Configuration for episode rollout recording and visualization."""

    # Recording control
    record_rollouts: bool = False
    rollout_dir: Path = Path("logs/rollouts")
    rollout_frequency: int = 0  # Record every N episodes (0 = disabled)
    rollout_max: int | None = 10  # Maximum rollouts to record (None = unlimited)
    rollout_include_unsolved: bool = False  # Whether to record failed episodes

    # GIF generation (note: disabled in trainer, use scripts/render_holdout_rollouts.py)
    rollout_make_gif: bool = False
    rollout_gif_duration: int = 140  # Frame duration in milliseconds

    def __post_init__(self):
        """Validate configuration."""
        if self.rollout_frequency < 0:
            raise ValueError(
                f"rollout_frequency must be non-negative, got {self.rollout_frequency}"
            )
        if self.rollout_max is not None and self.rollout_max < 0:
            raise ValueError(f"rollout_max must be non-negative if set, got {self.rollout_max}")
        if self.rollout_gif_duration <= 0:
            raise ValueError(
                f"rollout_gif_duration must be positive, got {self.rollout_gif_duration}"
            )
