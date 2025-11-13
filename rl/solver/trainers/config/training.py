"""Core DQN training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    """Core DQN training hyperparameters and optimization settings."""

    # Episode and batch settings
    episodes: int = 500
    batch_size: int = 64
    buffer_size: int = 100_000

    # Learning parameters
    gamma: float = 0.99
    lr: float = 1e-4
    target_update: int = 500
    grad_clip: float | None = 0.5

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000
    epsilon_schedule: str = "linear"  # {"linear", "exp"}
    epsilon_linear_steps: int | None = None  # If None, uses episodes

    # Prioritized Experience Replay (PER)
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 1e-4

    # Expert demonstration mixing (DQfD-style)
    expert_buffer_size: int = 0  # 0 = disabled
    expert_sample_ratio: float = 0.0

    # Network architecture
    use_dueling: bool = False

    # Performance optimizations
    use_amp: bool = False  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 1

    # Reward processing
    reward_scale: float = 1.0
    reward_clamp: float | None = 5.0

    # Logging
    use_tensorboard: bool = True
    log_every: int = 0  # 0 = only log evaluations

    # Environment
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    env_backend: str = "env2"  # Environment backend (only env2 supported)
    env2_reward: str = "potential"  # Reward preset for env2
    env2_channels: tuple[str, ...] | None = None  # Custom observation channels
    env2_undo_penalty: float = -0.1  # Penalty for undo actions
    steps_per_episode: int | None = None  # Override max steps per episode

    # Reward components (used by build_env and run_episode)
    move_penalty: float = -0.05
    distance_bonus: float = 0.4
    unsolved_penalty: float = -2.0
    dead_pocket_penalty: float = 0.0
    invalid_penalty: float = -0.5
    disconnect_penalty: float = -0.08
    degree_penalty: float = -0.08
    complete_color_bonus: float = 3.0
    complete_sustain_bonus: float = 0.0  # DISABLED: Overcomplicated
    complete_revert_penalty: float | None = None  # DISABLED: Overcomplicated
    solve_bonus: float = 20.0
    constraint_free_bonus: float = 0.0  # DISABLED: Redundant with constraint penalties
    solve_efficiency_bonus: float = 0.5  # NEW: Bonus per step remaining when solved
    segment_connection_bonus: float = 0.0
    path_extension_bonus: float = 0.0
    move_reduction_bonus: float = 0.0
    dead_end_penalty: float = 0.0
    loop_penalty: float = 0.0
    loop_window: int = 0
    progress_bonus: float = 0.0
    simple_rewards: bool = False  # Use simplified reward structure

    def __post_init__(self):
        """Validate configuration."""
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.buffer_size < self.batch_size:
            raise ValueError(
                f"buffer_size ({self.buffer_size}) must be >= batch_size ({self.batch_size})"
            )
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epsilon_schedule not in {"linear", "exp"}:
            raise ValueError(
                f"epsilon_schedule must be 'linear' or 'exp', got {self.epsilon_schedule}"
            )
        if self.expert_buffer_size > 0 and not 0.0 <= self.expert_sample_ratio <= 1.0:
            raise ValueError(
                f"expert_sample_ratio must be in [0, 1], got {self.expert_sample_ratio}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
