"""Rollout recorder for saving episode trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rl.solver.trainers.config import RolloutConfig
    from rl.solver.trainers.dqn import PuzzleConfig


class RolloutRecorder:
    """Records and saves episode rollouts (frames, metadata, optional GIFs).

    Handles:
    - Deciding when to record episodes
    - Saving frames to JSONL files
    - Saving metadata to JSON files
    - Tracking rollout count limits
    """

    def __init__(self, config: RolloutConfig, log_dir: Path):
        """Initialize rollout recorder.

        Args:
            config: Rollout configuration
            log_dir: Training run log directory
        """
        self.config = config
        self.rollout_dir: Path | None = None
        self.rollouts_recorded = 0

        if config.record_rollouts:
            self.rollout_dir = config.rollout_dir / log_dir.name
            self.rollout_dir.mkdir(parents=True, exist_ok=True)

    def should_record_training(self, episode: int) -> bool:
        """Check if training episode should be recorded.

        Args:
            episode: Current episode number

        Returns:
            True if episode should be recorded
        """
        if not self.config.record_rollouts:
            return False
        if self.config.rollout_frequency <= 0:
            return False
        if self.config.rollout_max is not None and self.rollouts_recorded >= self.config.rollout_max:
            return False
        return episode % self.config.rollout_frequency == 0

    def should_save(self, solved: bool) -> bool:
        """Check if rollout should be saved based on solve status.

        Args:
            solved: Whether episode was solved

        Returns:
            True if rollout should be saved
        """
        if self.config.rollout_max is not None and self.rollouts_recorded >= self.config.rollout_max:
            return False
        return solved or self.config.rollout_include_unsolved

    def save_rollout(
        self,
        frames: list[str],
        config: PuzzleConfig,
        episode: int,
        solved: bool,
        total_reward: float,
        tag: str,
        reward_breakdown: dict[str, float] | None = None,
        action_debug_info: list[dict[str, Any]] | None = None,
    ) -> tuple[Path, Path | None, Path]:
        """Save rollout frames and metadata.

        Args:
            frames: List of board state strings
            config: Puzzle configuration
            episode: Episode number
            solved: Whether episode was solved
            total_reward: Total episode reward
            tag: Tag for filename (e.g., "train", "eval")
            reward_breakdown: Optional reward component breakdown
            action_debug_info: Optional detailed action info per step

        Returns:
            Tuple of (jsonl_path, gif_path, meta_path)
        """
        if not self.rollout_dir:
            raise ValueError("Rollout directory not initialized (record_rollouts=False)")

        # Import here to avoid circular dependency
        from rl.solver.trainers.dqn import save_rollout_frames

        jsonl_path, gif_path, meta_path = save_rollout_frames(
            frames,
            self.rollout_dir,
            config,
            episode=episode,
            solved=solved,
            total_reward=total_reward,
            tag=tag,
            make_gif=self.config.rollout_make_gif,
            gif_duration=self.config.rollout_gif_duration,
            board_idx=config.board_idx,
            reward_breakdown=reward_breakdown,
            action_debug_info=action_debug_info,
        )

        self.rollouts_recorded += 1
        return jsonl_path, gif_path, meta_path

    def get_rollout_count(self) -> int:
        """Get number of rollouts recorded so far.

        Returns:
            Rollout count
        """
        return self.rollouts_recorded
