"""Backward compatibility helpers for DQN trainer refactor."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from rl.solver.core import RunLogger
    from rl.solver.reward_settings import RewardSettings
    from rl.solver.trainers.config import (
        CurriculumConfig,
        EvaluationConfig,
        RolloutConfig,
        TrainingConfig,
    )
    from rl.solver.trainers.dqn import DQNTrainingConfig


def map_old_config_to_new(
    cfg: DQNTrainingConfig,
) -> tuple[TrainingConfig, EvaluationConfig, CurriculumConfig, RolloutConfig, RewardSettings]:
    """Map old monolithic DQNTrainingConfig to new modular configs.

    Args:
        cfg: Old monolithic configuration

    Returns:
        Tuple of (training_config, evaluation_config, curriculum_config, rollout_config, reward_settings)
    """
    from rl.solver.reward_settings import RewardSettings, get_simple_reward_settings
    from rl.solver.trainers.config import (
        CurriculumConfig,
        EvaluationConfig,
        RolloutConfig,
        TrainingConfig,
    )

    # Training configuration
    training_config = TrainingConfig(
        episodes=cfg.episodes,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        gamma=cfg.gamma,
        lr=cfg.lr,
        target_update=cfg.target_update,
        grad_clip=cfg.grad_clip,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_schedule=cfg.epsilon_schedule,
        epsilon_linear_steps=cfg.epsilon_linear_steps,
        use_per=cfg.use_per,
        per_alpha=cfg.per_alpha,
        per_beta=cfg.per_beta,
        per_beta_increment=cfg.per_beta_increment,
        expert_buffer_size=cfg.expert_buffer_size,
        expert_sample_ratio=cfg.expert_sample_ratio,
        use_dueling=cfg.use_dueling,
        use_amp=cfg.use_amp,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        reward_scale=cfg.reward_scale,
        reward_clamp=cfg.reward_clamp,
        use_tensorboard=cfg.use_tensorboard,
        log_every=cfg.log_every,
        seed=cfg.seed,
        device=cfg.device,
        env_backend=cfg.env_backend,
        env2_reward=cfg.env2_reward,
        env2_channels=cfg.env2_channels,
        env2_undo_penalty=cfg.env2_undo_penalty,
        steps_per_episode=cfg.steps_per_episode,
        move_penalty=cfg.move_penalty,
        distance_bonus=cfg.distance_bonus,
        unsolved_penalty=cfg.unsolved_penalty,
        dead_pocket_penalty=cfg.dead_pocket_penalty,
        invalid_penalty=cfg.invalid_penalty,
        disconnect_penalty=cfg.disconnect_penalty,
        degree_penalty=cfg.degree_penalty,
        complete_color_bonus=cfg.complete_color_bonus,
        complete_sustain_bonus=cfg.complete_sustain_bonus,
        complete_revert_penalty=cfg.complete_revert_penalty,
        solve_bonus=cfg.solve_bonus,
        constraint_free_bonus=cfg.constraint_free_bonus,
        solve_efficiency_bonus=cfg.solve_efficiency_bonus,
        segment_connection_bonus=cfg.segment_connection_bonus,
        path_extension_bonus=cfg.path_extension_bonus,
        move_reduction_bonus=cfg.move_reduction_bonus,
        dead_end_penalty=cfg.dead_end_penalty,
        loop_penalty=cfg.loop_penalty,
        loop_window=cfg.loop_window,
        progress_bonus=cfg.progress_bonus,
        simple_rewards=cfg.simple_rewards,
    )

    # Evaluation configuration
    evaluation_config = EvaluationConfig(
        eval_interval=cfg.eval_interval,
        eval_episodes=cfg.eval_episodes,
        eval_epsilon=cfg.eval_epsilon,
        validation_csv=cfg.validation_csv,
        validation_limit=cfg.validation_limit,
        validation_eval_episodes=cfg.validation_eval_episodes,
    )

    # Curriculum configuration
    curriculum_config = CurriculumConfig(
        curriculum_six_prob_start=cfg.curriculum_six_prob_start,
        curriculum_six_prob_end=cfg.curriculum_six_prob_end,
        curriculum_six_prob_episodes=cfg.curriculum_six_prob_episodes,
        penalty_warmup=cfg.penalty_warmup,
        unsolved_penalty_start=cfg.unsolved_penalty_start,
        unsolved_penalty_warmup=cfg.unsolved_penalty_warmup,
    )

    # Rollout configuration
    rollout_config = RolloutConfig(
        record_rollouts=cfg.record_rollouts,
        rollout_dir=cfg.rollout_dir,
        rollout_frequency=cfg.rollout_frequency,
        rollout_max=cfg.rollout_max,
        rollout_include_unsolved=cfg.rollout_include_unsolved,
        rollout_make_gif=cfg.rollout_make_gif,
        rollout_gif_duration=cfg.rollout_gif_duration,
    )

    # Reward settings
    if cfg.simple_rewards:
        reward_settings = get_simple_reward_settings()
    else:
        reward_settings = RewardSettings(
            move_penalty=cfg.move_penalty,
            distance_bonus=cfg.distance_bonus,
            invalid_penalty=cfg.invalid_penalty,
            distance_penalty=getattr(cfg, "distance_penalty", 0.0),
            cell_fill_bonus=getattr(cfg, "cell_fill_bonus", 0.0),
            color_switch_penalty=getattr(cfg, "color_switch_penalty", 0.0),
            streak_bonus_per_move=getattr(cfg, "streak_bonus_per_move", 0.0),
            streak_length=getattr(cfg, "streak_length", 2),
            unsolved_penalty=cfg.unsolved_penalty,
            complete_color_bonus=cfg.complete_color_bonus,
            solve_bonus=cfg.solve_bonus,
            dead_pocket_penalty=cfg.dead_pocket_penalty,
            disconnect_penalty=cfg.disconnect_penalty,
            degree_penalty=cfg.degree_penalty,
            segment_connection_bonus=cfg.segment_connection_bonus,
            path_extension_bonus=cfg.path_extension_bonus,
            move_reduction_bonus=cfg.move_reduction_bonus,
            dead_end_penalty=cfg.dead_end_penalty,
        )

    return training_config, evaluation_config, curriculum_config, rollout_config, reward_settings


def run_training_with_new_trainer(
    cfg: DQNTrainingConfig,
    *,
    logger: RunLogger | None = None,
    eval_callback=None,
) -> tuple[Path, float, float | None]:
    """Run DQN training using new modular trainer (backward compatibility wrapper).

    Args:
        cfg: Old monolithic configuration
        logger: Optional MLflow logger
        eval_callback: Optional callback(episode, success_rate, avg_reward)

    Returns:
        Tuple of (output_path, best_train_success, best_validation_success)
    """
    from rl.solver.trainers.dqn_trainer import DQNTrainer

    # Map old config to new configs
    training_config, evaluation_config, curriculum_config, rollout_config, reward_settings = (
        map_old_config_to_new(cfg)
    )

    # Create trainer
    trainer = DQNTrainer(
        training_config=training_config,
        evaluation_config=evaluation_config,
        curriculum_config=curriculum_config,
        rollout_config=rollout_config,
        reward_settings=reward_settings,
        puzzle_csv=cfg.puzzle_csv,
        output=cfg.output,
        log_root=cfg.log_root,
        puzzle_limit=cfg.puzzle_limit,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        max_colors=cfg.max_colors,
        validation_csv=cfg.validation_csv,
        validation_limit=cfg.validation_limit,
        policy_init=cfg.policy_init,
        log_dir=cfg.log_dir,
        run_logger=logger,
    )

    # Run training
    return trainer.train(eval_callback=eval_callback)
