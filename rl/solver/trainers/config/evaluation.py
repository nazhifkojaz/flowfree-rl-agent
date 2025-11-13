"""Evaluation and validation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation during training."""

    # Training set evaluation
    eval_interval: int = 50  # Evaluate every N episodes (0 = disabled)
    eval_episodes: int = 5  # Number of episodes to run per evaluation
    eval_epsilon: float = 0.0  # Epsilon for evaluation (0 = greedy)

    # Validation set evaluation (optional)
    validation_csv: Path | None = None
    validation_limit: int | None = None  # Max puzzles from validation CSV
    validation_eval_episodes: int | None = None  # If None, use len(validation_configs)

    def __post_init__(self):
        """Validate configuration."""
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval must be non-negative, got {self.eval_interval}")
        if self.eval_episodes <= 0:
            raise ValueError(f"eval_episodes must be positive, got {self.eval_episodes}")
        if not 0.0 <= self.eval_epsilon <= 1.0:
            raise ValueError(f"eval_epsilon must be in [0, 1], got {self.eval_epsilon}")
        if self.validation_limit is not None and self.validation_limit <= 0:
            raise ValueError(
                f"validation_limit must be positive if set, got {self.validation_limit}"
            )
        if self.validation_eval_episodes is not None and self.validation_eval_episodes <= 0:
            raise ValueError(
                f"validation_eval_episodes must be positive if set, got {self.validation_eval_episodes}"
            )
