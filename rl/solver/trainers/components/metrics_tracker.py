"""Metrics tracking with rolling windows and CSV logging."""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.solver.trainers.dqn import PuzzleConfig


class MetricsTracker:
    """Tracks training metrics with rolling windows and CSV logging.

    Handles:
    - Rolling window statistics (100-episode averages)
    - CSV logging for per-episode and summary metrics
    - Best score tracking (train and validation)
    """

    def __init__(self, log_dir: Path):
        """Initialize metrics tracker.

        Args:
            log_dir: Directory for writing CSV files
        """
        self.log_dir = log_dir
        self.metrics_path = log_dir / "metrics.csv"
        self.summary_path = log_dir / "metrics_summary.csv"

        # Rolling windows for 100-episode averages
        self.reward_window: deque[float] = deque(maxlen=100)
        self.success_window: deque[int] = deque(maxlen=100)
        self.constraint_window: deque[float] = deque(maxlen=100)

        # Best scores
        self.best_train = 0.0
        self.best_validation: float | None = None
        self.best_primary = -1.0

        self._init_csv_files()

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                "episode,total_reward,steps,epsilon,avg_loss,solved,constraint_penalty,"
                "constraint_dead_pocket,constraint_disconnect,constraint_degree,"
                "eval_success,eval_reward,val_success,val_reward,board_size,color_count,"
                "reward_breakdown\n"
            )

        with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                "episode,rolling_reward_100,rolling_success_100,rolling_constraint_penalty_100\n"
            )

    def record_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        solved: bool,
        epsilon: float,
        avg_loss: float,
        constraint_penalty: float,
        constraint_counts: Counter[str],
        reward_breakdown: dict[str, float],
        config: PuzzleConfig,
        eval_success: str = "",
        eval_reward: str = "",
        val_success: str = "",
        val_reward: str = "",
    ) -> dict[str, float]:
        """Record episode metrics to CSV files.

        Args:
            episode: Episode number
            reward: Total episode reward
            steps: Number of steps taken
            solved: Whether episode was solved
            epsilon: Epsilon value used
            avg_loss: Average TD loss
            constraint_penalty: Total constraint penalty
            constraint_counts: Count of each constraint violation type
            reward_breakdown: Breakdown of reward components
            config: Puzzle configuration
            eval_success: Evaluation success rate (optional)
            eval_reward: Evaluation average reward (optional)
            val_success: Validation success rate (optional)
            val_reward: Validation average reward (optional)

        Returns:
            Dictionary with rolling averages
        """
        # Update rolling windows
        self.reward_window.append(reward)
        self.success_window.append(1 if solved else 0)
        self.constraint_window.append(constraint_penalty)

        # Compute rolling averages
        rolling_reward = sum(self.reward_window) / len(self.reward_window)
        rolling_success = sum(self.success_window) / len(self.success_window)
        rolling_constraint = sum(self.constraint_window) / len(self.constraint_window)

        # Write summary (rolling averages)
        with self.summary_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                f"{episode},{rolling_reward:.4f},{rolling_success:.4f},{rolling_constraint:.4f}\n"
            )

        # Write detailed metrics
        dead_hits = constraint_counts.get("dead_pocket", 0)
        disconnect_hits = constraint_counts.get("disconnect", 0)
        degree_hits = constraint_counts.get("degree", 0)
        reward_json = json.dumps(reward_breakdown, sort_keys=True).replace('"', '""')

        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                f"{episode},{reward:.4f},{steps},{epsilon:.4f},{avg_loss:.6f},{int(solved)},"
                f"{constraint_penalty:.4f},{dead_hits},{disconnect_hits},{degree_hits},"
                f"{eval_success},{eval_reward},{val_success},{val_reward},"
                f'{config.width},{config.color_count},"{reward_json}"\n'
            )

        return {
            "rolling_reward": rolling_reward,
            "rolling_success": rolling_success,
            "rolling_constraint": rolling_constraint,
        }

    def update_best_scores(
        self,
        train_success: float,
        val_success: float | None = None,
    ) -> tuple[bool, float]:
        """Update best scores and return whether new best was achieved.

        Args:
            train_success: Training set success rate
            val_success: Validation set success rate (optional)

        Returns:
            Tuple of (is_new_best, primary_metric)
            - is_new_best: True if this is the best score seen
            - primary_metric: The primary metric value (validation if available, else train)
        """
        self.best_train = max(self.best_train, train_success)

        primary_metric = train_success
        if val_success is not None:
            if self.best_validation is None or val_success > self.best_validation:
                self.best_validation = val_success
            primary_metric = val_success

        is_new_best = primary_metric > self.best_primary
        if is_new_best:
            self.best_primary = primary_metric

        return is_new_best, primary_metric

    def get_best_scores(self) -> dict[str, float | None]:
        """Get best scores achieved so far.

        Returns:
            Dictionary with best_train, best_validation, best_primary
        """
        return {
            "best_train": self.best_train,
            "best_validation": self.best_validation,
            "best_primary": self.best_primary,
        }
