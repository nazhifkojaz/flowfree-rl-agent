from __future__ import annotations

from .dqn import (
    DQNTrainingConfig,  # DEPRECATED
    collect_policy_rollout,
    compute_td_loss,
    epsilon_by_step,
    evaluate_policy as evaluate_dqn_policy,
    load_puzzle_configs as load_dqn_configs,
    run_episode,
    run_training as run_dqn_training,  # DEPRECATED
    save_rollout_frames,
    select_action,
)

# New modular API (recommended)
from .dqn_trainer import DQNTrainer

__all__ = [
    # Legacy API (deprecated)
    "DQNTrainingConfig",
    "run_dqn_training",
    # Core utilities (still used)
    "load_dqn_configs",
    "evaluate_dqn_policy",
    "run_episode",
    "collect_policy_rollout",
    "save_rollout_frames",
    "epsilon_by_step",
    "select_action",
    "compute_td_loss",
    # New modular API (recommended)
    "DQNTrainer",
]
