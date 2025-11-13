"""Curriculum learning configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning and difficulty progression."""

    # Size-based curriculum (5x5 → 6x6 progression)
    curriculum_six_prob_start: float = 0.5  # Initial probability of sampling 6x6 boards
    curriculum_six_prob_end: float = 0.85  # Final probability of sampling 6x6 boards
    curriculum_six_prob_episodes: int = 1500  # Episodes over which to ramp up

    # Penalty warmup (gradually increase constraint penalties)
    penalty_warmup: int = 200  # Episodes to reach full penalty strength
    unsolved_penalty_start: float | None = 0.0  # Starting unsolved penalty (None = use full)
    unsolved_penalty_warmup: int = 500  # Episodes to reach full unsolved penalty

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.curriculum_six_prob_start <= 1.0:
            raise ValueError(
                f"curriculum_six_prob_start must be in [0, 1], got {self.curriculum_six_prob_start}"
            )
        if not 0.0 <= self.curriculum_six_prob_end <= 1.0:
            raise ValueError(
                f"curriculum_six_prob_end must be in [0, 1], got {self.curriculum_six_prob_end}"
            )
        if self.curriculum_six_prob_episodes < 0:
            raise ValueError(
                f"curriculum_six_prob_episodes must be non-negative, got {self.curriculum_six_prob_episodes}"
            )
        if self.penalty_warmup < 0:
            raise ValueError(f"penalty_warmup must be non-negative, got {self.penalty_warmup}")
        if self.unsolved_penalty_warmup < 0:
            raise ValueError(
                f"unsolved_penalty_warmup must be non-negative, got {self.unsolved_penalty_warmup}"
            )
