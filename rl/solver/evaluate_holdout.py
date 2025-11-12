#!/usr/bin/env python3
"""Evaluate trained DQN policy on holdout test set.

This script evaluates a trained model on the test set and optionally records
rollouts with frame-by-frame logs and GIF visualizations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from rl.solver.constants import MAX_CHANNELS
from rl.solver.policies.q_network import FlowQNetwork
from rl.solver.reward_settings import RewardSettings, get_simple_reward_settings
from rl.solver.trainers.dqn import (
    DQNTrainingConfig,
    evaluate_policy,
    load_puzzle_configs,
    save_rollout_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN policy on holdout test set"
    )

    # I/O
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--test-csv", type=Path, default=Path("data/dqn_test.csv"), help="Test dataset CSV")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output metrics CSV")

    # Environment
    parser.add_argument("--min-size", type=int, default=5, help="Minimum board size")
    parser.add_argument("--max-size", type=int, default=5, help="Maximum board size")
    parser.add_argument("--max-colors", type=int, default=10, help="Maximum colors")
    parser.add_argument("--env2-reward", type=str, default="potential", help="Reward preset name")
    parser.add_argument("--env2-channels", nargs="+", default=["occupancy", "endpoints", "heads", "free", "congestion", "distance"], help="Observation channels")
    parser.add_argument("--simple-rewards", action="store_true", help="Use simple reward settings")
    parser.add_argument("--steps-per-episode", type=int, default=None, help="Max steps per episode")

    # Epsilon
    parser.add_argument("--epsilon", type=float, default=0.0, help="Evaluation epsilon (default: 0.0 for deterministic)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Epsilon start (for config)")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Epsilon end (for config)")
    parser.add_argument("--epsilon-schedule", type=str, default="linear", choices=["linear", "exp"], help="Epsilon schedule")
    parser.add_argument("--epsilon-linear-steps", type=int, default=None, help="Linear epsilon decay steps")

    # Rewards
    parser.add_argument("--move-penalty", type=float, default=-0.05, help="Move penalty")
    parser.add_argument("--distance-bonus", type=float, default=0.35, help="Distance bonus scale")
    parser.add_argument("--complete-bonus", type=float, default=1.8, help="Color completion bonus")
    parser.add_argument("--complete-sustain-bonus", type=float, default=0.1, help="Sustain bonus per completed color")
    parser.add_argument("--complete-revert-penalty", type=float, default=2.0, help="Penalty when undo reopens color")
    parser.add_argument("--solve-bonus", type=float, default=35.0, help="Solve bonus")
    parser.add_argument("--constraint-free-bonus", type=float, default=5.0, help="Constraint-free bonus")
    parser.add_argument("--unsolved-penalty", type=float, default=-2.0, help="Unsolved penalty")
    parser.add_argument("--unsolved-penalty-start", type=float, default=0.0, help="Starting unsolved penalty")
    parser.add_argument("--unsolved-penalty-warmup", type=int, default=500, help="Unsolved penalty warmup episodes")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Reward scale")
    parser.add_argument("--reward-clamp", type=float, default=5.0, help="Reward clamp")

    # Constraints
    parser.add_argument("--disconnect-penalty", type=float, default=-0.06, help="Disconnect penalty")
    parser.add_argument("--degree-penalty", type=float, default=-0.08, help="Degree penalty")
    parser.add_argument("--penalty-warmup", type=int, default=400, help="Constraint penalty warmup episodes")
    parser.add_argument("--undo-penalty", type=float, default=-0.25, help="Undo action penalty")

    # Loop detection
    parser.add_argument("--loop-penalty", type=float, default=-0.5, help="Loop penalty")
    parser.add_argument("--loop-window", type=int, default=6, help="Loop detection window")
    parser.add_argument("--progress-bonus", type=float, default=0.02, help="Progress bonus per new cell")

    # Rollouts
    parser.add_argument("--rollout-mode", type=str, default="none", choices=["none", "solved", "unsolved", "both"], help="Rollout recording mode")
    parser.add_argument("--rollout-dir", type=Path, default=Path("logs/holdout_rollouts"), help="Rollout output directory")
    parser.add_argument("--rollout-max", type=int, default=0, help="Max rollouts to record (0 = unlimited)")
    parser.add_argument("--gif", action="store_true", help="Generate GIFs for rollouts")
    parser.add_argument("--gif-duration", type=int, default=140, help="GIF frame duration (ms)")
    parser.add_argument("--rollout-tag", type=str, default="holdout", help="Tag for rollout filenames")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Rollout mode: {args.rollout_mode}")

    # Load model
    state_dict = torch.load(args.model_path, map_location=device)
    use_dueling = any(key.startswith("value_head.") for key in state_dict.keys())
    policy_net = FlowQNetwork(in_channels=MAX_CHANNELS, use_dueling=use_dueling).to(device)
    policy_net.load_state_dict(state_dict, strict=True)
    policy_net.eval()

    print(f"Loaded model (dueling={use_dueling})")

    # Load test configs
    reward_cfg = get_simple_reward_settings() if args.simple_rewards else RewardSettings()
    test_configs = load_puzzle_configs(
        args.test_csv,
        limit=None,
        reward_cfg=reward_cfg,
        seed=args.seed,
        min_size=args.min_size,
        max_size=args.max_size,
        max_colors=args.max_colors,
    )

    print(f"Loaded {len(test_configs)} test puzzles")

    # Build evaluation config
    cfg = DQNTrainingConfig(puzzle_csv=Path("data/dqn_train.csv"))
    cfg = replace(
        cfg,
        env_backend="env2",
        env2_reward=args.env2_reward,
        env2_channels=tuple(args.env2_channels),
        simple_rewards=args.simple_rewards,
        min_size=args.min_size,
        max_size=args.max_size,
        max_colors=args.max_colors,
        steps_per_episode=args.steps_per_episode,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_schedule=args.epsilon_schedule,
        epsilon_linear_steps=args.epsilon_linear_steps,
        move_penalty=args.move_penalty,
        distance_bonus=args.distance_bonus,
        complete_color_bonus=args.complete_bonus,
        complete_sustain_bonus=args.complete_sustain_bonus,
        complete_revert_penalty=args.complete_revert_penalty,
        solve_bonus=args.solve_bonus,
        constraint_free_bonus=args.constraint_free_bonus,
        unsolved_penalty=args.unsolved_penalty,
        unsolved_penalty_start=args.unsolved_penalty_start,
        unsolved_penalty_warmup=args.unsolved_penalty_warmup,
        reward_scale=args.reward_scale,
        reward_clamp=args.reward_clamp,
        disconnect_penalty=args.disconnect_penalty,
        degree_penalty=args.degree_penalty,
        penalty_warmup=args.penalty_warmup,
        env2_undo_penalty=args.undo_penalty,
        loop_penalty=args.loop_penalty,
        loop_window=args.loop_window,
        progress_bonus=args.progress_bonus,
    )

    # Rollout recording setup
    record_rollouts = args.rollout_mode != "none"
    rollout_callback: Callable | None = None

    if record_rollouts:
        args.rollout_dir.mkdir(parents=True, exist_ok=True)
        saved_rollouts = {"count": 0}

        def should_record(solved: bool) -> bool:
            if args.rollout_mode == "solved":
                return solved
            if args.rollout_mode == "unsolved":
                return not solved
            if args.rollout_mode == "both":
                return True
            return False

        def _rollout_callback(config, frames, actions, solved_flag, total_reward, breakdown, action_debug_info=None):
            if not should_record(solved_flag):
                return
            if args.rollout_max and saved_rollouts["count"] >= args.rollout_max:
                return
            save_rollout_frames(
                frames,
                args.rollout_dir,
                config,
                episode=saved_rollouts["count"] + 1,
                solved=solved_flag,
                total_reward=total_reward,
                tag=args.rollout_tag,
                make_gif=args.gif,
                gif_duration=args.gif_duration,
                board_idx=config.board_idx,
                reward_breakdown=breakdown,
                action_debug_info=action_debug_info,
            )
            saved_rollouts["count"] += 1

        rollout_callback = _rollout_callback
        print(f"Rollout recording enabled: {args.rollout_dir}")

    # Evaluate
    success_rate, avg_reward, eval_details = evaluate_policy(
        policy_net,
        test_configs,
        device=device,
        episodes=len(test_configs),
        seed=args.seed,
        epsilon=args.epsilon,
        cfg=cfg,
        record_details=True,
        record_rollouts=record_rollouts,
        rollout_callback=rollout_callback,
    )

    print(f"\n{'='*60}")
    print(f"Evaluated {len(test_configs)} puzzles")
    print(f"Test Success Rate: {success_rate:.2%}")
    print(f"Test Average Reward: {avg_reward:.2f}")
    print(f"{'='*60}\n")

    # Save metrics CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "board_idx",
        "board_size",
        "color_count",
        "puzzle",
        "steps",
        "reward",
        "solved",
        "terminated",
        "truncated",
        "constraint_penalty",
        "constraint_violations",
        "loop_penalty",
        "progress_bonus",
    ]

    with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_details or []:
            writer.writerow(row)

    print(f"Metrics saved to: {args.output_csv}")

    if record_rollouts:
        print(f"Rollouts saved to: {args.rollout_dir} (count={saved_rollouts['count']})")


if __name__ == "__main__":
    main()
