"""Unified logger for MLflow, TensorBoard, and console output."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.solver.core import RunLogger


class DQNLogger:
    """Unified logger for MLflow, TensorBoard, and console.

    Eliminates code duplication by providing a single interface that
    logs to both MLflow and TensorBoard simultaneously.
    """

    def __init__(
        self,
        run_logger: RunLogger,
        log_dir: Path,
        use_tensorboard: bool = True,
    ):
        """Initialize unified logger.

        Args:
            run_logger: MLflow run logger (or NullRunLogger)
            log_dir: Directory for TensorBoard logs
            use_tensorboard: Whether to enable TensorBoard logging
        """
        self.run_logger = run_logger
        self.log_dir = log_dir

        # Initialize TensorBoard writer
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
                print(f"✓ TensorBoard logging enabled: {log_dir / 'tensorboard'}")
            except ImportError:
                print("⚠ TensorBoard not available (torch.utils.tensorboard not found)")
        else:
            print("✓ TensorBoard disabled")

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        epsilon: float,
        avg_loss: float,
        solved: bool,
        constraint_penalty: float,
        buffer_size: int,
        reward_breakdown: dict[str, float],
    ) -> None:
        """Log episode metrics to both MLflow and TensorBoard.

        Args:
            episode: Episode number
            total_reward: Total episode reward
            steps: Number of steps taken
            epsilon: Epsilon value used
            avg_loss: Average TD loss
            solved: Whether episode was solved
            constraint_penalty: Total constraint penalty
            buffer_size: Current replay buffer size
            reward_breakdown: Reward component breakdown
        """
        # MLflow logging
        self.run_logger.log_metric("train/total_reward", total_reward, episode)
        self.run_logger.log_metric("train/avg_loss", avg_loss, episode)
        self.run_logger.log_metric("train/epsilon", epsilon, episode)
        self.run_logger.log_metric("train/solved", float(int(solved)), episode)
        self.run_logger.log_metric("train/constraint_penalty", constraint_penalty, episode)
        self.run_logger.log_metric("train/buffer_size", float(buffer_size), episode)

        # Log reward breakdown components
        for component, value in reward_breakdown.items():
            self.run_logger.log_metric(f"reward_components/{component}", value, episode)

        # TensorBoard logging (if enabled)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/total_reward", total_reward, episode)
            self.tb_writer.add_scalar("train/avg_loss", avg_loss, episode)
            self.tb_writer.add_scalar("train/epsilon", epsilon, episode)
            self.tb_writer.add_scalar("train/solved", float(int(solved)), episode)
            self.tb_writer.add_scalar("train/constraint_penalty", constraint_penalty, episode)
            self.tb_writer.add_scalar("train/buffer_size", float(buffer_size), episode)
            self.tb_writer.add_scalar("train/steps", steps, episode)

            # Log reward breakdown to TensorBoard
            for component, value in reward_breakdown.items():
                self.tb_writer.add_scalar(f"reward_components/{component}", value, episode)

    def log_evaluation(
        self,
        episode: int,
        train_success: float,
        train_reward: float,
        val_success: float | None = None,
        val_reward: float | None = None,
    ) -> None:
        """Log evaluation metrics to both MLflow and TensorBoard.

        Args:
            episode: Episode number
            train_success: Training set success rate
            train_reward: Training set average reward
            val_success: Validation set success rate (optional)
            val_reward: Validation set average reward (optional)
        """
        # MLflow logging
        self.run_logger.log_metric("eval/success_rate", train_success, episode)
        self.run_logger.log_metric("eval/avg_reward", train_reward, episode)

        if val_success is not None:
            self.run_logger.log_metric("val/success_rate", val_success, episode)
        if val_reward is not None:
            self.run_logger.log_metric("val/avg_reward", val_reward, episode)

        # TensorBoard logging (if enabled)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("eval/success_rate", train_success, episode)
            self.tb_writer.add_scalar("eval/avg_reward", train_reward, episode)

            if val_success is not None:
                self.tb_writer.add_scalar("val/success_rate", val_success, episode)
            if val_reward is not None:
                self.tb_writer.add_scalar("val/avg_reward", val_reward, episode)

    def log_warning(self, name: str, value: float, episode: int) -> None:
        """Log warning metric (e.g., high reward without solving).

        Args:
            name: Metric name (e.g., "warn/high_reward_unsolved")
            value: Metric value
            episode: Episode number
        """
        self.run_logger.log_metric(name, value, episode)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, episode)

    def log_final(
        self,
        episodes: int,
        best_train: float,
        best_validation: float | None,
        buffer_size: int,
    ) -> None:
        """Log final training metrics.

        Args:
            episodes: Total number of episodes
            best_train: Best training success rate
            best_validation: Best validation success rate (optional)
            buffer_size: Final replay buffer size
        """
        primary_best = best_validation if best_validation is not None else best_train

        self.run_logger.log_metric("final/best_success_rate", primary_best, episodes)
        self.run_logger.log_metric("final/best_train_success", best_train, episodes)
        if best_validation is not None:
            self.run_logger.log_metric("final/best_validation_success", best_validation, episodes)
        self.run_logger.log_metric("final/episodes", float(episodes), episodes)
        self.run_logger.log_metric("final/buffer_size", float(buffer_size), episodes)

    def log_artifact(self, path: Path) -> None:
        """Log artifact file.

        Args:
            path: Path to artifact file
        """
        self.run_logger.log_artifact(path)

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        self.run_logger.close()
