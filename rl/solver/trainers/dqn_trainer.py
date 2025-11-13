"""DQN Trainer orchestrator class."""

from __future__ import annotations

import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from rl.env.config import default_max_steps
from rl.solver.constants import MAX_CHANNELS
from rl.solver.core import NullRunLogger, RunLogger
from rl.solver.data import ReplayBuffer
from rl.solver.policies.policy import FlowPolicy, load_policy
from rl.solver.policies.q_network import FlowQNetwork
from rl.solver.reward_settings import RewardSettings
from rl.solver.trainers.components import (
    CurriculumManager,
    DQNLogger,
    MetricsTracker,
    RolloutRecorder,
)
from rl.solver.trainers.config import (
    CurriculumConfig,
    EvaluationConfig,
    RolloutConfig,
    TrainingConfig,
)
from rl.solver.trainers.dqn import (
    PuzzleConfig,
    build_env,
    collect_policy_rollout,
    epsilon_by_step,
    evaluate_policy,
    load_puzzle_configs,
    run_episode,
)


def ensure_log_dir(log_root: Path) -> Path:
    """Create timestamped log directory.

    Args:
        log_root: Root directory for logs

    Returns:
        Created log directory path
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = log_root / f"dqn_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_hyperparams(log_dir: Path, params: dict) -> None:
    """Write hyperparameters to JSON file.

    Args:
        log_dir: Directory to write to
        params: Hyperparameter dictionary
    """
    import json

    path = log_dir / "hyperparams.json"
    path.write_text(json.dumps(params, indent=2))


class DQNTrainer:
    """Main DQN training orchestrator.

    Coordinates components to train a DQN agent with:
    - Curriculum learning
    - Prioritized experience replay
    - Expert demonstration mixing
    - Rollout recording
    - Multi-backend logging (MLflow + TensorBoard)
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        evaluation_config: EvaluationConfig,
        curriculum_config: CurriculumConfig,
        rollout_config: RolloutConfig,
        reward_settings: RewardSettings,
        puzzle_csv: Path,
        output: Path,
        log_root: Path,
        puzzle_limit: int | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        max_colors: int | None = None,
        validation_csv: Path | None = None,
        validation_limit: int | None = None,
        policy_init: Path | None = None,
        log_dir: Path | None = None,
        run_logger: RunLogger | None = None,
    ):
        """Initialize DQN trainer.

        Args:
            training_config: Training hyperparameters (includes steps_per_episode)
            evaluation_config: Evaluation settings
            curriculum_config: Curriculum learning settings
            rollout_config: Rollout recording settings
            reward_settings: Reward shaping configuration
            puzzle_csv: Path to training puzzle CSV
            output: Path to save final model
            log_root: Root directory for logs
            puzzle_limit: Max training puzzles to load
            min_size: Minimum board size filter
            max_size: Maximum board size filter
            max_colors: Maximum color count filter
            validation_csv: Path to validation puzzle CSV
            validation_limit: Max validation puzzles to load
            policy_init: Path to supervised warm-start checkpoint
            log_dir: Override log directory (if None, creates timestamped)
            run_logger: MLflow logger (if None, uses NullRunLogger)
        """
        self.training_config = training_config
        self.evaluation_config = evaluation_config
        self.curriculum_config = curriculum_config
        self.rollout_config = rollout_config
        self.reward_settings = reward_settings
        self.output = output

        # Setup random seeds
        self._setup_random_seeds()
        self.device = torch.device(training_config.device)

        # Load puzzles
        self.train_configs = load_puzzle_configs(
            puzzle_csv,
            limit=puzzle_limit,
            min_size=min_size,
            max_size=max_size,
            max_colors=max_colors,
            reward_cfg=reward_settings,
            seed=training_config.seed,
        )
        if not self.train_configs:
            raise ValueError("No training puzzles available after filtering")

        self.val_configs: list[PuzzleConfig] = []
        if validation_csv is not None:
            self.val_configs = load_puzzle_configs(
                validation_csv,
                limit=validation_limit,
                min_size=min_size,
                max_size=max_size,
                max_colors=max_colors,
                reward_cfg=reward_settings,
                seed=training_config.seed + 131,
            )

        # Setup log directory
        self.log_dir = log_dir if log_dir is not None else ensure_log_dir(log_root)
        self._write_hyperparams()

        # Initialize components
        self.curriculum_manager = CurriculumManager(
            self.train_configs,
            curriculum_config,
            training_config.seed,
        )

        self.metrics_tracker = MetricsTracker(self.log_dir)

        self.rollout_recorder = RolloutRecorder(rollout_config, self.log_dir)

        self.logger = DQNLogger(
            run_logger or NullRunLogger(),
            self.log_dir,
            use_tensorboard=training_config.use_tensorboard,
        )

        # Initialize networks
        self.policy_net, self.target_net = self._build_networks(policy_init)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=training_config.lr)

        # Initialize replay buffers
        self.buffer = ReplayBuffer(
            training_config.buffer_size,
            alpha=training_config.per_alpha,
            beta=training_config.per_beta,
            beta_increment=training_config.per_beta_increment,
        )

        self.expert_buffer: ReplayBuffer | None = None
        if training_config.expert_buffer_size > 0 and training_config.expert_sample_ratio > 0:
            self.expert_buffer = ReplayBuffer(
                training_config.expert_buffer_size,
                alpha=0.0,
                beta=0.0,
                beta_increment=0.0,
            )

        # Initialize gradient scaler for AMP
        self.scaler: GradScaler | None = None
        if training_config.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            print("✓ Automatic Mixed Precision (AMP) enabled")

        if training_config.gradient_accumulation_steps > 1:
            effective_batch = training_config.batch_size * training_config.gradient_accumulation_steps
            print(
                f"✓ Gradient Accumulation enabled: {training_config.gradient_accumulation_steps} steps "
                f"(effective batch={effective_batch})"
            )

        # Log parameters to MLflow
        self._log_params()

        # Track global step and accumulation counter
        self.global_step = 0
        self.accumulation_counter = 0

    def train(
        self,
        eval_callback: Callable[[int, float, float], None] | None = None,
    ) -> tuple[Path, float, float | None]:
        """Run full training loop.

        Args:
            eval_callback: Optional callback(episode, success_rate, avg_reward)

        Returns:
            Tuple of (output_path, best_train_success, best_validation_success)
        """
        try:
            for episode in range(1, self.training_config.episodes + 1):
                self._train_episode(episode)

                # Periodic evaluation
                if self._should_evaluate(episode):
                    self._run_evaluation(episode, eval_callback)

            # Save final model and log final metrics
            self._finalize_training()

        finally:
            self.logger.close()

        return (
            self.output,
            self.metrics_tracker.best_train,
            self.metrics_tracker.best_validation,
        )

    def _train_episode(self, episode: int) -> None:
        """Run single training episode.

        Args:
            episode: Current episode number
        """
        # Select puzzle via curriculum
        config = self.curriculum_manager.select_puzzle(episode)
        config = self._apply_episode_config(config, episode)

        # Build environment
        env = build_env(self.training_config, config)

        # Compute epsilon
        epsilon = self._compute_epsilon(episode)

        # Setup frame recording if needed
        record_frames: list[str] | None = None
        if self.rollout_recorder.should_record_training(episode):
            record_frames = []

        # Run episode
        (
            ep_reward,
            ep_steps,
            solved,
            losses,
            self.global_step,
            constraint_penalty,
            constraint_counts,
            reward_breakdown,
            self.accumulation_counter,
        ) = run_episode(
            env,
            self.policy_net,
            cfg=self.training_config,
            epsilon=epsilon,
            device=self.device,
            rng=random,
            buffer=self.buffer,
            optimizer=self.optimizer,
            target_net=self.target_net,
            gamma=self.training_config.gamma,
            batch_size=self.training_config.batch_size,
            grad_clip=self.training_config.grad_clip,
            global_step=self.global_step,
            reward_scale=self.training_config.reward_scale,
            reward_clamp=self.training_config.reward_clamp,
            record_frames=record_frames,
            expert_buffer=self.expert_buffer,
            scaler=self.scaler,
            accumulation_counter=self.accumulation_counter,
        )

        # Record curriculum result
        self.curriculum_manager.record_result(config, solved)

        # Update target network
        if self.global_step % max(1, self.training_config.target_update) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Track metrics
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.metrics_tracker.record_episode(
            episode=episode,
            reward=ep_reward,
            steps=ep_steps,
            solved=solved,
            epsilon=epsilon,
            avg_loss=avg_loss,
            constraint_penalty=constraint_penalty,
            constraint_counts=constraint_counts,
            reward_breakdown=reward_breakdown,
            config=config,
        )

        # Save rollout if needed
        if record_frames and self.rollout_recorder.should_save(solved):
            jsonl_path, gif_path, meta_path = self.rollout_recorder.save_rollout(
                record_frames,
                config,
                episode,
                solved,
                ep_reward,
                "train",
                reward_breakdown=reward_breakdown,
            )
            print(f"[rollout] Saved training episode {episode} to {jsonl_path}")
            self.logger.log_artifact(jsonl_path)
            self.logger.log_artifact(meta_path)
            if gif_path is not None:
                self.logger.log_artifact(gif_path)

        # Log to MLflow + TensorBoard
        self.logger.log_episode(
            episode=episode,
            total_reward=ep_reward,
            steps=ep_steps,
            epsilon=epsilon,
            avg_loss=avg_loss,
            solved=solved,
            constraint_penalty=constraint_penalty,
            buffer_size=len(self.buffer),
            reward_breakdown=reward_breakdown,
        )

        # Console logging
        if self.training_config.log_every > 0 and episode % self.training_config.log_every == 0:
            print(
                f"[train] episode={episode}/{self.training_config.episodes} reward={ep_reward:.3f} "
                f"steps={ep_steps} epsilon={epsilon:.3f} avg_loss={avg_loss:.4f} "
                f"solved={int(solved)} buffer={len(self.buffer)}"
            )

        # Check for high-reward unsolved (potential bug indicator)
        self._check_high_reward_unsolved(episode, config, ep_reward, ep_steps, solved)

    def _run_evaluation(
        self,
        episode: int,
        eval_callback: Callable[[int, float, float], None] | None,
    ) -> None:
        """Run evaluation on train and validation sets.

        Args:
            episode: Current episode number
            eval_callback: Optional callback for evaluation results
        """
        # Evaluate on training set
        train_success, train_reward, _ = evaluate_policy(
            self.policy_net,
            self.train_configs,
            device=self.device,
            episodes=self.evaluation_config.eval_episodes,
            seed=self.training_config.seed + episode,
            epsilon=self.evaluation_config.eval_epsilon,
            cfg=self.training_config,
            record_details=False,
        )

        # Evaluate on validation set if available
        val_success: float | None = None
        val_reward: float | None = None
        if self.val_configs:
            val_episodes = self.evaluation_config.validation_eval_episodes or len(self.val_configs)
            if val_episodes > 0:
                val_success, val_reward, _ = evaluate_policy(
                    self.policy_net,
                    self.val_configs,
                    device=self.device,
                    episodes=val_episodes,
                    seed=self.training_config.seed + episode + 10_000,
                    epsilon=self.evaluation_config.eval_epsilon,
                    cfg=self.training_config,
                    record_details=False,
                )

        # Log evaluation metrics
        self.logger.log_evaluation(
            episode=episode,
            train_success=train_success,
            train_reward=train_reward,
            val_success=val_success,
            val_reward=val_reward,
        )

        # Check for new best
        is_new_best, primary_metric = self.metrics_tracker.update_best_scores(
            train_success,
            val_success,
        )

        if is_new_best:
            # Save best checkpoint
            torch.save(self.policy_net.state_dict(), self.log_dir / "best.pt")

            # Record best evaluation rollout
            if (
                self.rollout_config.record_rollouts
                and self.rollout_recorder.get_rollout_count() < (self.rollout_config.rollout_max or float("inf"))
            ):
                eval_pool = self.val_configs if val_success is not None and self.val_configs else self.train_configs
                eval_config = random.choice(eval_pool)
                frames, eval_solved, eval_total_reward = collect_policy_rollout(
                    self.policy_net,
                    eval_config,
                    device=self.device,
                    epsilon=self.evaluation_config.eval_epsilon,
                    seed=self.training_config.seed + episode + 1,
                    cfg=self.training_config,
                )

                if eval_solved or self.rollout_config.rollout_include_unsolved:
                    jsonl_path, gif_path, meta_path = self.rollout_recorder.save_rollout(
                        frames,
                        eval_config,
                        episode,
                        eval_solved,
                        eval_total_reward,
                        "eval",
                    )
                    print(f"[rollout] Saved evaluation snapshot at episode {episode} to {jsonl_path}")
                    self.logger.log_artifact(jsonl_path)
                    self.logger.log_artifact(meta_path)
                    if gif_path is not None:
                        self.logger.log_artifact(gif_path)

        # Console output
        msg = f"[eval] episode={episode} train_success={train_success:.3f} train_reward={train_reward:.3f}"
        if val_success is not None and val_reward is not None:
            msg += f" val_success={val_success:.3f} val_reward={val_reward:.3f}"
        msg += f" buffer={len(self.buffer)}"
        print(msg)

        # Invoke callback if provided
        if eval_callback is not None:
            eval_callback(episode, train_success, train_reward)

    def _finalize_training(self) -> None:
        """Finalize training: save model, log final metrics, log artifacts."""
        # Save final model
        torch.save(self.policy_net.state_dict(), self.output)

        # Log final metrics
        self.logger.log_final(
            episodes=self.training_config.episodes,
            best_train=self.metrics_tracker.best_train,
            best_validation=self.metrics_tracker.best_validation,
            buffer_size=len(self.buffer),
        )

        # Log artifacts
        self.logger.log_artifact(self.metrics_tracker.metrics_path)
        if (self.log_dir / "best.pt").exists():
            self.logger.log_artifact(self.log_dir / "best.pt")
        if self.output.exists():
            self.logger.log_artifact(self.output)

    def _setup_random_seeds(self) -> None:
        """Initialize random seeds for reproducibility."""
        torch.manual_seed(self.training_config.seed)
        random.seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)

    def _build_networks(self, policy_init: Path | None) -> tuple[FlowQNetwork, FlowQNetwork]:
        """Build policy and target networks.

        Args:
            policy_init: Optional path to supervised warm-start checkpoint

        Returns:
            Tuple of (policy_net, target_net)
        """
        policy_net = FlowQNetwork(
            in_channels=MAX_CHANNELS,
            use_dueling=self.training_config.use_dueling,
        ).to(self.device)

        # Load supervised warm-start if provided
        if policy_init is not None and policy_init.exists():
            supervised_policy = FlowPolicy(in_channels=MAX_CHANNELS)
            load_policy(supervised_policy, policy_init, map_location=self.device)
            policy_net.backbone.load_state_dict(
                supervised_policy.backbone.state_dict(),
                strict=False,
            )
            print(f"Loaded supervised backbone weights from {policy_init}")

        # Create target network as copy of policy network
        target_net = FlowQNetwork(
            in_channels=MAX_CHANNELS,
            use_dueling=self.training_config.use_dueling,
        ).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        return policy_net, target_net

    def _apply_episode_config(self, config: PuzzleConfig, episode: int) -> PuzzleConfig:
        """Apply episode-specific configuration overrides.

        Args:
            config: Base puzzle configuration
            episode: Current episode number

        Returns:
            Modified puzzle configuration
        """
        # Override step cap if requested
        if self.training_config.steps_per_episode is not None:
            config = replace(config, max_steps=self.training_config.steps_per_episode)

        # Apply curriculum-based penalty warmup
        penalty = self._current_unsolved_penalty(episode)
        penalty_scale = self._current_penalty_scale(episode)

        env_reward = replace(
            config.reward,
            unsolved_penalty=penalty,
            disconnect_penalty=config.reward.disconnect_penalty * penalty_scale,
            degree_penalty=config.reward.degree_penalty * penalty_scale,
        )

        return replace(config, reward=env_reward)

    def _compute_epsilon(self, episode: int) -> float:
        """Compute epsilon for epsilon-greedy exploration.

        Args:
            episode: Current episode number

        Returns:
            Epsilon value in [epsilon_end, epsilon_start]
        """
        schedule = self.training_config.epsilon_schedule.lower()

        if schedule == "linear":
            return epsilon_by_step(
                episode,
                start=self.training_config.epsilon_start,
                end=self.training_config.epsilon_end,
                decay=self.training_config.epsilon_decay,
                schedule="linear",
                linear_total=self.training_config.epsilon_linear_steps or self.training_config.episodes,
            )
        else:  # exponential
            return epsilon_by_step(
                self.global_step,
                start=self.training_config.epsilon_start,
                end=self.training_config.epsilon_end,
                decay=self.training_config.epsilon_decay,
                schedule="exp",
            )

    def _current_unsolved_penalty(self, episode: int) -> float:
        """Compute current unsolved penalty with warmup.

        Args:
            episode: Current episode number

        Returns:
            Unsolved penalty value
        """
        unsolved_start = self.curriculum_config.unsolved_penalty_start
        if unsolved_start is None:
            unsolved_start = self.reward_settings.unsolved_penalty

        warmup_total = max(0, self.curriculum_config.unsolved_penalty_warmup)
        if warmup_total == 0 or self.reward_settings.unsolved_penalty == unsolved_start:
            return self.reward_settings.unsolved_penalty

        progress = min(1.0, max(0.0, episode / warmup_total))
        return unsolved_start + progress * (self.reward_settings.unsolved_penalty - unsolved_start)

    def _current_penalty_scale(self, episode: int) -> float:
        """Compute current constraint penalty scale with warmup.

        Args:
            episode: Current episode number

        Returns:
            Penalty scale in [0.0, 1.0]
        """
        if self.curriculum_config.penalty_warmup <= 0:
            return 1.0
        return min(1.0, max(0.0, episode / self.curriculum_config.penalty_warmup))

    def _should_evaluate(self, episode: int) -> bool:
        """Check if evaluation should run this episode.

        Args:
            episode: Current episode number

        Returns:
            True if evaluation should run
        """
        if self.evaluation_config.eval_interval <= 0:
            return False
        return episode % self.evaluation_config.eval_interval == 0

    def _check_high_reward_unsolved(
        self,
        episode: int,
        config: PuzzleConfig,
        reward: float,
        steps: int,
        solved: bool,
    ) -> None:
        """Check for anomalous high reward without solving (potential bug).

        Args:
            episode: Current episode number
            config: Puzzle configuration
            reward: Episode reward
            steps: Steps taken
            solved: Whether puzzle was solved
        """
        if solved:
            return

        target_steps = config.max_steps or default_max_steps(config.width, config.height)
        solve_bonus = getattr(self.reward_settings, "solve_bonus", 20.0)

        high_reward_unsolved = (
            target_steps > 0
            and steps >= 0.9 * target_steps
            and reward >= max(solve_bonus * 0.5, 5.0)
        )

        if high_reward_unsolved:
            self.logger.log_warning("warn/high_reward_unsolved", reward, episode)
            print(
                f"[warn] episode {episode} reward={reward:.3f} steps={steps} solved={solved} "
                f"(near max {target_steps})"
            )

    def _write_hyperparams(self) -> None:
        """Write all hyperparameters to JSON file."""
        params = {
            **asdict(self.training_config),
            **asdict(self.evaluation_config),
            **asdict(self.curriculum_config),
            **asdict(self.rollout_config),
            "puzzle_count": len(self.train_configs),
            "validation_count": len(self.val_configs),
            "output": str(self.output),
        }
        # Convert Path objects to strings
        params = {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()}
        write_hyperparams(self.log_dir, params)

    def _log_params(self) -> None:
        """Log parameters to MLflow."""
        params = {
            **asdict(self.training_config),
            **asdict(self.evaluation_config),
            **asdict(self.curriculum_config),
            **asdict(self.rollout_config),
            "puzzle_count": len(self.train_configs),
            "validation_count": len(self.val_configs),
        }
        # Convert Path objects to strings
        params = {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()}
        self.logger.run_logger.log_params(params)
        self.logger.run_logger.set_tags({"phase": "dqn_finetune"})
