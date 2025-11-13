"""Curriculum manager for adaptive puzzle selection."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.solver.trainers.dqn import PuzzleConfig
    from rl.solver.trainers.config import CurriculumConfig


class CurriculumManager:
    """Manages puzzle selection based on success rates and curriculum progression.

    Implements:
    - Color-based curriculum (prioritize colors with lower success rates)
    - Size-based curriculum (gradual 5x5 → 6x6 progression)
    - Per-puzzle weighting (prioritize puzzles with lower success rates)
    """

    def __init__(
        self,
        configs: list[PuzzleConfig],
        curriculum_config: CurriculumConfig,
        seed: int,
    ):
        """Initialize curriculum manager.

        Args:
            configs: List of puzzle configurations to sample from
            curriculum_config: Curriculum settings
            seed: Random seed for reproducibility
        """
        self.configs = configs
        self.curriculum_config = curriculum_config
        self.rng = random.Random(seed)

        # Build buckets for stratified sampling
        self.size_buckets: dict[int, list[PuzzleConfig]] = defaultdict(list)
        self.color_buckets: dict[int, list[PuzzleConfig]] = defaultdict(list)
        for cfg in configs:
            self.size_buckets[cfg.width].append(cfg)
            self.color_buckets[cfg.color_count].append(cfg)

        # Track episode counts and successes for adaptive weighting
        self.color_episode_counts: Counter[int] = Counter()
        self.color_success_counts: Counter[int] = Counter()
        self.puzzle_episode_counts: Counter[str] = Counter()
        self.puzzle_success_counts: Counter[str] = Counter()

    def select_puzzle(self, episode: int) -> PuzzleConfig:
        """Select next puzzle based on curriculum and success rates.

        Strategy:
        1. If color buckets exist, use color-based curriculum
        2. Otherwise, use size-based curriculum (5x5 vs 6x6 ramp-up)
        3. Within each bucket, weight by inverse success rate

        Args:
            episode: Current episode number (used for curriculum progression)

        Returns:
            Selected puzzle configuration
        """
        # Color-based curriculum (preferred if available)
        available_colors = [c for c, bucket in self.color_buckets.items() if bucket]
        if available_colors:
            weights = [self._color_weight(c) for c in available_colors]
            chosen_color = self.rng.choices(available_colors, weights=weights, k=1)[0]
            return self._choose_from_bucket(self.color_buckets[chosen_color])

        # Size-based curriculum (5x5 vs 6x6)
        bucket5 = self.size_buckets.get(5, [])
        bucket6 = self.size_buckets.get(6, [])
        if bucket5 and bucket6:
            prob_six = self._six_probability(episode)
            if self.rng.random() < prob_six:
                return self._choose_from_bucket(bucket6)
            return self._choose_from_bucket(bucket5)

        # Fallback: uniform sampling from all configs
        return self._choose_from_bucket(self.configs)

    def record_result(self, config: PuzzleConfig, solved: bool) -> None:
        """Record episode result for curriculum tracking.

        Args:
            config: Puzzle configuration that was attempted
            solved: Whether the puzzle was solved
        """
        self.color_episode_counts[config.color_count] += 1
        if solved:
            self.color_success_counts[config.color_count] += 1

        if config.board_idx:
            self.puzzle_episode_counts[config.board_idx] += 1
            if solved:
                self.puzzle_success_counts[config.board_idx] += 1

    def _color_weight(self, color: int) -> float:
        """Compute sampling weight for color based on success rate.

        Higher weight = lower success rate = more difficult.

        Args:
            color: Color count

        Returns:
            Sampling weight in [0.05, 1.0]
        """
        total = self.color_episode_counts[color]
        if total == 0:
            return 1.0
        success = self.color_success_counts[color]
        success_rate = success / total
        return max(0.05, 1.0 - success_rate)

    def _puzzle_weight(self, config: PuzzleConfig) -> float:
        """Compute sampling weight for specific puzzle.

        Args:
            config: Puzzle configuration

        Returns:
            Sampling weight in [0.05, 1.0]
        """
        if not config.board_idx:
            return 1.0
        total = self.puzzle_episode_counts[config.board_idx]
        if total == 0:
            return 1.0
        success = self.puzzle_success_counts[config.board_idx]
        rate = success / total
        return max(0.05, 1.0 - rate)

    def _six_probability(self, episode: int) -> float:
        """Compute probability of selecting 6x6 puzzle based on curriculum.

        Linearly increases from curriculum_six_prob_start to curriculum_six_prob_end
        over curriculum_six_prob_episodes.

        Args:
            episode: Current episode number

        Returns:
            Probability in [0.0, 1.0]
        """
        duration = max(0, self.curriculum_config.curriculum_six_prob_episodes)
        if duration == 0:
            return self.curriculum_config.curriculum_six_prob_end

        progress = min(1.0, max(0.0, episode / duration))
        return self.curriculum_config.curriculum_six_prob_start + progress * (
            self.curriculum_config.curriculum_six_prob_end
            - self.curriculum_config.curriculum_six_prob_start
        )

    def _choose_from_bucket(self, bucket: list[PuzzleConfig]) -> PuzzleConfig:
        """Choose puzzle from bucket with success-rate weighting.

        Args:
            bucket: List of puzzle configurations

        Returns:
            Selected puzzle configuration
        """
        if not bucket:
            return self.rng.choice(self.configs)
        weights = [self._puzzle_weight(cfg) for cfg in bucket]
        return self.rng.choices(bucket, weights=weights, k=1)[0]
