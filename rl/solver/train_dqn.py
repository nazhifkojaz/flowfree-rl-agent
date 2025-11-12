from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl.solver.core.logging import NullRunLogger
from rl.solver.trainers.dqn import (
    DQNTrainingConfig,
    collect_policy_rollout,
    compute_td_loss,
    ensure_log_dir,
    evaluate_policy,
    load_puzzle_configs,
    run_episode,
    run_training,
    save_rollout_frames,
    select_action,
    write_hyperparams,
    epsilon_by_step,
)


def parse_args(argv: list[str] | None = None) -> DQNTrainingConfig:
    parser = argparse.ArgumentParser(description="DQN fine-tuning for the FlowFree RL agent")
    parser.add_argument("--puzzle-csv", type=Path, default=Path("flowfree/training_data.csv"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-update", type=int, default=500, help="Frequency (in steps) to update the target network")
    parser.add_argument("--eval-interval", type=int, default=50, help="Run evaluation every N episodes")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=10_000)
    parser.add_argument(
        "--epsilon-schedule",
        choices=("linear", "exp"),
        default="linear",
        help="Exploration decay schedule (linear anneal by default).",
    )
    parser.add_argument(
        "--epsilon-linear-steps",
        type=int,
        default=None,
        help="Total episodes over which to anneal epsilon when using the linear schedule (defaults to --episodes).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None, help="Training device (default: auto-detect)")
    parser.add_argument("--log-root", type=Path, default=Path("logs/dqn"))
    parser.add_argument("--output", type=Path, default=Path("models/rl_dqn.pt"))
    parser.add_argument("--puzzle-limit", type=int, default=None, help="Optional cap on puzzles loaded from CSV")
    parser.add_argument("--min-size", type=int, default=None, help="Smallest board size to include")
    parser.add_argument("--max-size", type=int, default=None, help="Largest board size to include")
    parser.add_argument("--max-colors", type=int, default=None, help="Skip puzzles above this color count")
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=None,
        help="Optional hard cap on steps per episode (default: board_area + 10)",
    )
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clipping value (None to disable)")
    parser.add_argument("--log-every", type=int, default=0, help="Log training progress every N episodes (0 to disable)")
    parser.add_argument("--policy-init", type=Path, default=None, help="Optional supervised policy initialisation")
    parser.add_argument("--move-penalty", type=float, default=-0.05)
    parser.add_argument("--distance-bonus", type=float, default=0.35)
    parser.add_argument("--unsolved-penalty", type=float, default=-2.0)
    parser.add_argument("--unsolved-penalty-start", type=float, default=0.0)
    parser.add_argument("--unsolved-penalty-warmup", type=int, default=500)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clamp", type=float, default=5.0)
    parser.add_argument("--dead-pocket-penalty", type=float, default=0.0)
    parser.add_argument("--invalid-penalty", type=float, default=-0.5)
    parser.add_argument("--disconnect-penalty", type=float, default=-0.08)
    parser.add_argument("--degree-penalty", type=float, default=-0.08)
    parser.add_argument("--complete-bonus", type=float, default=1.0)
    parser.add_argument("--complete-sustain-bonus", type=float, default=0.0)
    parser.add_argument(
        "--complete-revert-penalty",
        type=float,
        default=None,
        help="Penalty applied when undoing a previously completed colour (env2 only).",
    )
    parser.add_argument("--solve-bonus", type=float, default=25.0)
    parser.add_argument("--constraint-free-bonus", type=float, default=5.0)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--segment-connection-bonus", type=float, default=0.0)
    parser.add_argument("--path-extension-bonus", type=float, default=0.0)
    parser.add_argument("--move-reduction-bonus", type=float, default=0.0)
    parser.add_argument("--dead-end-penalty", type=float, default=0.0)
    parser.add_argument("--penalty-warmup", type=int, default=200)
    parser.add_argument("--curriculum-six-prob-start", type=float, default=0.5)
    parser.add_argument("--curriculum-six-prob-end", type=float, default=0.85)
    parser.add_argument("--curriculum-six-prob-episodes", type=int, default=1500)
    parser.add_argument("--record-rollouts", action="store_true")
    parser.add_argument("--rollout-dir", type=Path, default=Path("logs/dqn/rollouts"))
    parser.add_argument("--rollout-frequency", type=int, default=0)
    parser.add_argument("--rollout-max", type=int, default=10)
    parser.add_argument("--rollout-include-unsolved", action="store_true")
    parser.add_argument("--rollout-make-gif", action="store_true")
    parser.add_argument("--rollout-gif-duration", type=int, default=140)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--per-beta-increment", type=float, default=1e-4)
    parser.add_argument("--use-per", action="store_true")
    parser.add_argument("--use-dueling", action="store_true")
    parser.add_argument("--expert-buffer-size", type=int, default=0, help="Size of optional expert replay buffer (0 disables)")
    parser.add_argument("--expert-sample-ratio", type=float, default=0.0, help="Fraction of each batch drawn from expert buffer")
    parser.add_argument(
        "--simple-rewards",
        action="store_true",
        help="Use simplified reward structure (potential-based only, removes reward hacking signals)",
    )
    parser.add_argument(
        "--env-backend",
        choices=("env2",),
        default="env2",
        help="Select environment implementation (env2 only).",
    )
    parser.add_argument(
        "--env2-reward",
        default="potential",
        help="Reward preset to use with env2 (default: potential)",
    )
    parser.add_argument(
        "--env2-channels",
        nargs="+",
        default=None,
        help="Optional override for env2 observation channels (space-separated list)",
    )
    parser.add_argument(
        "--env2-undo-penalty",
        type=float,
        default=-0.1,
        help="Penalty applied when taking the undo action in env2 (negative values discourage overuse)",
    )
    parser.add_argument("--loop-penalty", type=float, default=0.0, help="Penalty applied when repeating recent board states")
    parser.add_argument("--loop-window", type=int, default=0, help="Window size for detecting repeated board states (0 disables)")
    parser.add_argument("--progress-bonus", type=float, default=0.0, help="Bonus per newly filled cell to encourage forward progress")

    args = parser.parse_args(argv)
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    rollout_max = None if args.rollout_max is not None and args.rollout_max <= 0 else args.rollout_max

    return DQNTrainingConfig(
        puzzle_csv=args.puzzle_csv,
        episodes=args.episodes,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        lr=args.lr,
        target_update=args.target_update,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        epsilon_schedule=args.epsilon_schedule,
        epsilon_linear_steps=args.epsilon_linear_steps,
        seed=args.seed,
        device=device,
        log_root=args.log_root,
        output=args.output,
        puzzle_limit=args.puzzle_limit,
        min_size=args.min_size,
        max_size=args.max_size,
        max_colors=args.max_colors,
        steps_per_episode=args.steps_per_episode,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        policy_init=args.policy_init,
        move_penalty=args.move_penalty,
        distance_bonus=args.distance_bonus,
        unsolved_penalty=args.unsolved_penalty,
        unsolved_penalty_start=args.unsolved_penalty_start,
        unsolved_penalty_warmup=args.unsolved_penalty_warmup,
        reward_scale=args.reward_scale,
        reward_clamp=args.reward_clamp,
        dead_pocket_penalty=args.dead_pocket_penalty,
        invalid_penalty=args.invalid_penalty,
        disconnect_penalty=args.disconnect_penalty,
        degree_penalty=args.degree_penalty,
        complete_color_bonus=args.complete_bonus,
        complete_sustain_bonus=args.complete_sustain_bonus,
        complete_revert_penalty=args.complete_revert_penalty,
        solve_bonus=args.solve_bonus,
        constraint_free_bonus=args.constraint_free_bonus,
        eval_epsilon=args.eval_epsilon,
        segment_connection_bonus=args.segment_connection_bonus,
        path_extension_bonus=args.path_extension_bonus,
        move_reduction_bonus=args.move_reduction_bonus,
        dead_end_penalty=args.dead_end_penalty,
        penalty_warmup=args.penalty_warmup,
        curriculum_six_prob_start=args.curriculum_six_prob_start,
        curriculum_six_prob_end=args.curriculum_six_prob_end,
        curriculum_six_prob_episodes=args.curriculum_six_prob_episodes,
        record_rollouts=args.record_rollouts,
        rollout_dir=args.rollout_dir,
        rollout_frequency=args.rollout_frequency,
        rollout_max=rollout_max,
        rollout_include_unsolved=args.rollout_include_unsolved,
        rollout_make_gif=args.rollout_make_gif,
        rollout_gif_duration=args.rollout_gif_duration,
        validation_csv=args.validation_csv,
        validation_limit=args.validation_limit,
        validation_eval_episodes=args.validation_episodes,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
        per_beta_increment=args.per_beta_increment,
        use_per=args.use_per,
        use_dueling=args.use_dueling,
        simple_rewards=args.simple_rewards,
        env_backend=args.env_backend,
        env2_reward=args.env2_reward,
        env2_channels=tuple(args.env2_channels) if args.env2_channels is not None else None,
        env2_undo_penalty=args.env2_undo_penalty,
        loop_penalty=args.loop_penalty,
        loop_window=args.loop_window,
        progress_bonus=args.progress_bonus,
        expert_buffer_size=args.expert_buffer_size,
        expert_sample_ratio=args.expert_sample_ratio,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    # Use MLflow logger for TensorBoard/MLflow tracking
    from rl.solver.core.logging import MLflowRunLogger
    logger = MLflowRunLogger(
        experiment_name="dqn_training",
        tracking_uri=f"file://{cfg.log_root.absolute()}/mlruns"
    )
    try:
        output_path, best_eval, best_val = run_training(cfg, logger=logger)
        summary = (
            f"best train/validation success={best_eval:.3f}/{best_val:.3f}"
            if best_val is not None
            else f"best eval success={best_eval:.3f}"
        )
        print(f"Training complete. Saved DQN to {output_path} ({summary})")
    finally:
        logger.close()


__all__ = [
    "DQNTrainingConfig",
    "epsilon_by_step",
    "select_action",
    "compute_td_loss",
    "load_puzzle_configs",
    "run_episode",
    "evaluate_policy",
    "collect_policy_rollout",
    "save_rollout_frames",
    "ensure_log_dir",
    "write_hyperparams",
    "run_training",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()
