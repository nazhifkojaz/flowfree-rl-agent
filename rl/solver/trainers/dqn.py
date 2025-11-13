"""DQN training utilities and legacy configuration.

Post-refactor organization:
- This file now contains SHARED UTILITIES used by the new DQNTrainer
- Core functions (run_episode, evaluate_policy, etc.) are still here
- DQNTrainingConfig is DEPRECATED - use new modular configs instead
- For new code, use: rl.solver.trainers.dqn_trainer.DQNTrainer

Maintained for backward compatibility with existing scripts.
"""

from __future__ import annotations

import csv
import json
import math
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from rl.env.config import (
    BoardShape,
    EnvConfig,
    DEFAULT_OBSERVATION,
    MaskConfig,
    ObservationSpec,
    RewardPreset,
    default_max_steps,
)
from rl.env.env import FlowFreeEnv
from rl.solver.constants import MAX_CHANNELS
from rl.solver.core import NullRunLogger, RunLogger
from rl.solver.data import ReplayBuffer, Transition
from rl.solver.observation import EncodedObservation, encode_observation, mask_to_tensor
from rl.solver.policies.policy import FlowPolicy, load_policy
from rl.solver.policies.q_network import FlowQNetwork
from rl.solver.reward_settings import RewardSettings


def _infer_color_count(head_mask: torch.Tensor) -> int:
    flat = head_mask.view(head_mask.shape[0], -1)
    active = (flat.sum(dim=1) > 0).nonzero(as_tuple=False)
    if active.numel() == 0:
        return 1
    return int(active[-1, 0].item() + 1)


@dataclass(frozen=True)
class PuzzleConfig:
    width: int
    height: int
    color_count: int
    puzzle: str
    max_steps: int
    reward: RewardSettings
    board_idx: str | None = None


@dataclass
class DQNTrainingConfig:
    """DEPRECATED: Monolithic DQN training configuration.

    This configuration is maintained for backward compatibility only.
    For new code, use the modular configuration classes:
        - rl.solver.trainers.config.TrainingConfig (core hyperparameters)
        - rl.solver.trainers.config.EvaluationConfig (evaluation settings)
        - rl.solver.trainers.config.CurriculumConfig (curriculum learning)
        - rl.solver.trainers.config.RolloutConfig (rollout recording)
        - rl.solver.reward_settings.RewardSettings (reward shaping)

    And use the DQNTrainer class directly:
        from rl.solver.trainers.dqn_trainer import DQNTrainer

    This config will be removed in a future version.
    """

    puzzle_csv: Path
    episodes: int = 500
    batch_size: int = 64
    buffer_size: int = 100_000
    gamma: float = 0.99
    lr: float = 1e-4  # Fixed from 2.096e-6 (too low for DQN)
    target_update: int = 500  # Reduced from 1000 for better stability with short episodes
    eval_interval: int = 50
    eval_episodes: int = 5
    epsilon_start: float = 1.0  # Fixed from 0.152 (should start with full exploration)
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000
    epsilon_schedule: str = "linear"  # {"linear", "exp"}
    epsilon_linear_steps: int | None = None
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_root: Path = Path("flowfree/logs/rl_training")
    log_dir: Path | None = None
    output: Path = Path("models/rl_dqn.pt")
    puzzle_limit: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    max_colors: int | None = None
    steps_per_episode: int | None = None
    grad_clip: float | None = 0.5
    log_every: int = 0
    policy_init: Path | None = None
    move_penalty: float = -0.05
    distance_bonus: float = 0.4
    unsolved_penalty: float = -2.0
    unsolved_penalty_start: float | None = 0.0
    unsolved_penalty_warmup: int = 500
    reward_scale: float = 1.0
    reward_clamp: float | None = 5.0
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
    eval_epsilon: float = 0.0
    segment_connection_bonus: float = 0.0
    path_extension_bonus: float = 0.0
    move_reduction_bonus: float = 0.0
    dead_end_penalty: float = 0.0
    penalty_warmup: int = 200
    curriculum_six_prob_start: float = 0.5
    curriculum_six_prob_end: float = 0.85
    curriculum_six_prob_episodes: int = 1500
    record_rollouts: bool = False
    rollout_dir: Path = Path("flowfree/logs/rl_training/rollouts")
    rollout_frequency: int = 0
    rollout_max: int | None = 10
    rollout_include_unsolved: bool = False
    rollout_make_gif: bool = False
    rollout_gif_duration: int = 140
    validation_csv: Path | None = None
    validation_limit: int | None = None
    validation_eval_episodes: int | None = None
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 1e-4
    use_per: bool = True
    use_dueling: bool = False
    simple_rewards: bool = False  # Use simplified reward structure
    env_backend: str = "env2"  # Only env2 is supported
    env2_reward: str = "potential"
    env2_channels: tuple[str, ...] | None = None
    env2_undo_penalty: float = -0.1
    expert_buffer_size: int = 0
    expert_sample_ratio: float = 0.0
    loop_penalty: float = 0.0
    loop_window: int = 0
    progress_bonus: float = 0.0
    use_amp: bool = False  # Enable Automatic Mixed Precision (faster on modern GPUs)
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps (effective batch size = batch_size * N)
    use_tensorboard: bool = True  # Enable TensorBoard logging (disable to save 1-2GB RAM)

    def __post_init__(self):
        """Emit deprecation warning when config is instantiated."""
        warnings.warn(
            "DQNTrainingConfig is deprecated and will be removed in a future version. "
            "Please use the modular config classes (TrainingConfig, EvaluationConfig, etc.) "
            "and DQNTrainer class instead. See rl/solver/trainers/dqn_trainer.py for examples.",
            DeprecationWarning,
            stacklevel=2,
        )


def epsilon_by_step(
    step: int,
    *,
    start: float,
    end: float,
    decay: int,
    schedule: str = "linear",
    linear_total: int | None = None,
) -> float:
    if schedule == "linear":
        total = linear_total if linear_total and linear_total > 0 else decay if decay > 0 else None
        if not total or total <= 0:
            return end
        frac = max(0.0, min(1.0, step / total))
        return start + (end - start) * frac

    if decay <= 0:
        return end
    return end + (start - end) * math.exp(-step / decay)


def select_action(
    q_network: FlowQNetwork,
    state: torch.Tensor,
    mask: torch.Tensor,
    head_mask: torch.Tensor,
    target_mask: torch.Tensor,
    color_count: int,
    *,
    epsilon: float,
    rng: random.Random,
) -> int:
    """Epsilon-greedy policy that respects the legal action mask."""

    valid_actions = torch.nonzero(mask > 0.0, as_tuple=False).flatten()
    if len(valid_actions) == 0:
        # BUG FIX: This should never happen after env.py fix
        # If it does, it indicates a bug in the environment's mask computation
        raise RuntimeError(
            f"No valid actions available! All {len(mask)} actions are masked. "
            "This should have been caught by the environment and forced truncation. "
            "Check env.py for the all-actions-masked detection logic."
        )

    if rng.random() < epsilon:
        idx = rng.randrange(len(valid_actions))
        return int(valid_actions[idx].item())

    with torch.no_grad():
        q_values = q_network(
            state.unsqueeze(0),
            head_masks=head_mask.unsqueeze(0),
            target_masks=target_mask.unsqueeze(0),
            color_counts=torch.tensor([color_count], device=state.device),
        ).squeeze(0)
        masked_q = q_values.clone()
        masked_q[mask <= 0.0] = -float('inf')
        action = int(torch.argmax(masked_q).item())
        if mask[action] <= 0.0:
            idx = rng.randrange(len(valid_actions))
            action = int(valid_actions[idx].item())
        return action


def compute_td_loss(
    policy_net: FlowQNetwork,
    target_net: FlowQNetwork,
    batch: Sequence[Transition],
    *,
    device: torch.device,
    gamma: float,
    reward_scale: float = 1.0,
    reward_clamp: float | None = None,
    importance_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.stack([transition.state for transition in batch]).to(device)
    head_masks = torch.stack([transition.head_mask for transition in batch]).to(device)
    target_masks = torch.stack([transition.target_mask for transition in batch]).to(device)
    color_counts = torch.tensor([transition.color_count for transition in batch], device=device)

    actions = torch.tensor([transition.action for transition in batch], device=device)
    rewards = torch.tensor([transition.reward for transition in batch], device=device, dtype=torch.float32)
    if reward_scale != 1.0:
        rewards = rewards * reward_scale
    if reward_clamp is not None:
        rewards = torch.clamp(rewards, -reward_clamp, reward_clamp)

    next_states = torch.stack([transition.next_state for transition in batch]).to(device)
    next_head_masks = torch.stack([transition.next_head_mask for transition in batch]).to(device)
    next_target_masks = torch.stack([transition.next_target_mask for transition in batch]).to(device)
    next_color_counts = torch.tensor([transition.next_color_count for transition in batch], device=device)
    masks = torch.stack([transition.mask for transition in batch]).to(device)
    next_masks = torch.stack([transition.next_mask for transition in batch]).to(device)
    dones = torch.tensor([transition.done for transition in batch], device=device, dtype=torch.bool)

    q_values = policy_net(
        states,
        head_masks=head_masks,
        target_masks=target_masks,
        color_counts=color_counts,
    )
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_policy_q = policy_net(
            next_states,
            head_masks=next_head_masks,
            target_masks=next_target_masks,
            color_counts=next_color_counts,
        )
        next_policy_q = next_policy_q.masked_fill(next_masks <= 0.0, -float("inf"))
        next_actions = torch.argmax(next_policy_q, dim=1)

        next_target_q = target_net(
            next_states,
            head_masks=next_head_masks,
            target_masks=next_target_masks,
            color_counts=next_color_counts,
        )
        next_target_q = next_target_q.masked_fill(next_masks <= 0.0, -float("inf"))
        next_best = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        has_valid = (next_masks > 0.0).any(dim=1)
        next_best = next_best.masked_fill(~has_valid, 0.0)
        next_best = next_best.masked_fill(dones, 0.0)
        targets = rewards + gamma * next_best

    td_errors = targets - q_selected

    if importance_weights is not None:
        loss = (importance_weights * F.smooth_l1_loss(q_selected, targets, reduction="none")).mean()
    else:
        loss = F.smooth_l1_loss(q_selected, targets)

    return loss, td_errors.detach()


def load_puzzle_configs(
    csv_path: Path,
    *,
    limit: int | None,
    min_size: int | None,
    max_size: int | None,
    max_colors: int | None,
    reward_cfg: RewardSettings,
    seed: int,
) -> list[PuzzleConfig]:
    filtered: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            width = int(row["BoardSize"])
            height = int(row.get("BoardHeight", width))
            color_count = int(row["ColorCount"])

            if min_size is not None and width < min_size:
                continue
            if max_size is not None and width > max_size:
                continue
            if max_colors is not None and color_count > max_colors:
                continue

            filtered.append(row)

    if not filtered:
        return []

    rng = random.Random(seed)
    rng.shuffle(filtered)

    if limit is not None:
        filtered = filtered[:limit]

    configs: list[PuzzleConfig] = []
    for row in filtered:
        width = int(row["BoardSize"])
        height = int(row.get("BoardHeight", width))
        color_count = int(row["ColorCount"])
        puzzle = row["InitialPuzzle"]

        config = PuzzleConfig(
            width=width,
            height=height,
            color_count=color_count,
            puzzle=puzzle,
            max_steps=default_max_steps(width, height),
            reward=reward_cfg,
            board_idx=row.get("board_idx"),
        )
        configs.append(config)

    return configs


def _build_env2_reward(cfg: "DQNTrainingConfig", puzzle_config: PuzzleConfig):
    reward = puzzle_config.reward
    params = {
        "move_penalty": reward.move_penalty,
        "distance_scale": reward.distance_bonus,
        "complete_bonus": reward.complete_color_bonus,
        "solve_bonus": reward.solve_bonus,
        "solve_efficiency_bonus": cfg.solve_efficiency_bonus,
        "invalid_penalty": cfg.invalid_penalty,
        "dead_pocket_penalty": reward.dead_pocket_penalty,
        "disconnect_penalty": reward.disconnect_penalty,
        "degree_penalty": reward.degree_penalty,
        "unsolved_penalty": reward.unsolved_penalty,
        "undo_penalty": cfg.env2_undo_penalty,
        "segment_connection_bonus": reward.segment_connection_bonus,
        "path_extension_bonus": reward.path_extension_bonus,
        "move_reduction_bonus": reward.move_reduction_bonus,
        "dead_end_penalty": reward.dead_end_penalty,
    }
    if cfg.complete_revert_penalty is not None:
        params["complete_revert_penalty"] = cfg.complete_revert_penalty
    if cfg.complete_sustain_bonus:
        params["complete_sustain_bonus"] = cfg.complete_sustain_bonus
    if cfg.loop_penalty:
        params["loop_penalty"] = cfg.loop_penalty
        params["loop_window"] = cfg.loop_window
    if cfg.progress_bonus:
        params["progress_bonus"] = cfg.progress_bonus
    if cfg.env2_reward != "potential":
        raise ValueError(f"Unsupported env2 reward preset '{cfg.env2_reward}'")
    return RewardPreset(
        name=cfg.env2_reward,
        components=("potential", "completion", "constraints"),
        params=params,
    )


def _build_env2_config(cfg: "DQNTrainingConfig", puzzle_config: PuzzleConfig):
    observation_spec = (
        DEFAULT_OBSERVATION
        if cfg.env2_channels is None
        else ObservationSpec(
            channels=tuple(cfg.env2_channels),
            dtype=DEFAULT_OBSERVATION.dtype,
            include_temporal_planes=DEFAULT_OBSERVATION.include_temporal_planes,
        )
    )
    reward_preset = _build_env2_reward(cfg, puzzle_config)
    return EnvConfig(
        shape=BoardShape(
            width=puzzle_config.width,
            height=puzzle_config.height,
            color_count=puzzle_config.color_count,
        ),
        puzzle=puzzle_config.puzzle,
        reward=reward_preset,
        observation=observation_spec,
        mask=MaskConfig(),
        max_steps=puzzle_config.max_steps,
        seed=cfg.seed,
    )


def build_env(cfg: "DQNTrainingConfig", puzzle_config: PuzzleConfig) -> FlowEnv:
    if cfg.env_backend != "env2":
        raise ValueError("Only env2 backend is supported.")

    from rl.env.env import FlowFreeEnv

    env2_config = _build_env2_config(cfg, puzzle_config)
    return FlowFreeEnv(env2_config)


def run_episode(
    env: FlowEnv,
    policy_net: FlowQNetwork,
    *,
    cfg: DQNTrainingConfig,
    epsilon: float,
    device: torch.device,
    rng: random.Random,
    buffer: ReplayBuffer | None = None,
    optimizer: optim.Optimizer | None = None,
    target_net: FlowQNetwork | None = None,
    gamma: float = 0.99,
    batch_size: int = 64,
    grad_clip: float | None = 1.0,
    global_step: int = 0,
    reward_scale: float = 1.0,
    reward_clamp: float | None = None,
    record_frames: list[str] | None = None,
    expert_buffer: ReplayBuffer | None = None,
    scaler: GradScaler | None = None,
    accumulation_counter: int = 0,
) -> tuple[float, int, bool, list[float], int, float, Counter[str], dict[str, float], int]:
    obs, _ = env.reset()
    if record_frames is not None:
        record_frames.append(env.board_string)
    encoded = encode_observation(obs, device=device)
    state = encoded.state
    head_mask = encoded.head_mask
    target_mask = encoded.target_mask
    color_count = encoded.color_count
    mask = mask_to_tensor(obs["action_mask"], device=device)

    total_reward = 0.0
    losses: list[float] = []
    steps = 0
    solved = False

    constraint_penalty_total = 0.0
    constraint_counts: Counter[str] = Counter()
    reward_breakdown: dict[str, float] = {}
    episode_breakdown: dict[str, float] = {}

    episode_transitions: list[Transition] | None = [] if expert_buffer is not None else None

    while True:
        if not torch.any(mask > 0.0):
            # No legal moves remain; treat as truncated to avoid looping on invalid actions.
            break
        action = select_action(
            policy_net,
            state,
            mask,
            head_mask,
            target_mask,
            color_count,
            epsilon=epsilon,
            rng=rng,
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        if terminated:
            solved = True
        if record_frames is not None:
            record_frames.append(env.board_string)

        if info:
            constraint_penalty_total += float(info.get("constraint_penalty", 0.0))
            for name in info.get("constraint_violations", []):
                constraint_counts[name] += 1

            # Accumulate reward breakdown for logging
            if "reward_breakdown" in info:
                for component, value in info["reward_breakdown"].items():
                    reward_breakdown[component] = reward_breakdown.get(component, 0.0) + value
                    episode_breakdown[component] = episode_breakdown.get(component, 0.0) + value

        next_encoded = encode_observation(next_obs, device=device)
        next_state = next_encoded.state
        next_head_mask = next_encoded.head_mask
        next_target_mask = next_encoded.target_mask
        next_color_count = next_encoded.color_count
        next_mask = mask_to_tensor(next_obs["action_mask"], device=device)

        if buffer is not None:
            transition = buffer.push(
                state,
                mask,
                head_mask,
                target_mask,
                color_count,
                action,
                reward,
                next_state,
                next_mask,
                next_head_mask,
                next_target_mask,
                next_color_count,
                done,
            )
            if episode_transitions is not None:
                episode_transitions.append(transition)

        if (
            buffer is not None
            and optimizer is not None
            and target_net is not None
            and len(buffer) >= batch_size
        ):
            expert_batch_size = 0
            expert_samples: list[Transition] = []
            if expert_buffer is not None and cfg.expert_sample_ratio > 0:
                desired = int(batch_size * cfg.expert_sample_ratio)
                desired = min(desired, batch_size - 1)
                if desired > 0 and len(expert_buffer) >= desired:
                    expert_samples, _, _ = expert_buffer.sample(desired, prioritized=False)
                    expert_batch_size = len(expert_samples)
            main_batch_size = batch_size - expert_batch_size
            batch, indices, weights = buffer.sample(main_batch_size, prioritized=cfg.use_per)
            if expert_batch_size:
                batch = batch + expert_samples
                if weights is not None:
                    expert_weights = np.ones(expert_batch_size, dtype=np.float64)
                    weights = np.concatenate([weights, expert_weights])
            # Compute loss (with optional AMP)
            with autocast(enabled=cfg.use_amp):
                loss, td_errors = compute_td_loss(
                    policy_net,
                    target_net,
                    batch,
                    device=device,
                    gamma=cfg.gamma,
                    reward_scale=cfg.reward_scale,
                    reward_clamp=cfg.reward_clamp,
                    importance_weights=torch.tensor(weights, device=device) if weights is not None else None,
                )

            if cfg.use_per and indices is not None:
                buffer.update_priorities(indices, td_errors[: len(indices)].detach().cpu().numpy())

            # Scale loss for gradient accumulation
            scaled_loss = loss / cfg.gradient_accumulation_steps

            # Backward pass (with optional AMP scaling)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accumulation_counter += 1

            # Update weights only after accumulating enough gradients
            if accumulation_counter % cfg.gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item()))

        state, mask = next_state, next_mask
        head_mask, target_mask = next_head_mask, next_target_mask
        color_count = next_color_count
        if done:
            break

    total_violations = sum(constraint_counts.values())
    if steps > 0 and cfg.constraint_free_bonus and total_violations == 0:
        total_reward += cfg.constraint_free_bonus
        reward_breakdown["constraint_free_bonus"] = reward_breakdown.get("constraint_free_bonus", 0.0) + cfg.constraint_free_bonus

    if solved and expert_buffer is not None and episode_transitions:
        for transition in episode_transitions:
            expert_buffer.push_transition(transition)

    return total_reward, steps, solved, losses, global_step + steps, constraint_penalty_total, constraint_counts, reward_breakdown, accumulation_counter


def evaluate_policy(
    policy_net: FlowQNetwork,
    configs: list[PuzzleConfig],
    *,
    device: torch.device,
    episodes: int,
    seed: int,
    epsilon: float,
    cfg: DQNTrainingConfig,
    record_details: bool = False,
    record_rollouts: bool = False,
    rollout_callback: Callable[[PuzzleConfig, list[str], list[int], bool, float, dict[str, float], list[dict[str, Any]]], None] | None = None,
) -> tuple[float, float, list[dict[str, Any]] | None]:
    if not configs:
        empty = [] if record_details else None
        return 0.0, 0.0, empty

    policy_net.eval()
    rng = random.Random(seed)
    total_reward = 0.0
    solved = 0
    details: list[dict[str, Any]] | None = [] if record_details else None

    for eval_idx in range(episodes):
        config = rng.choice(configs)
        if cfg.steps_per_episode is not None:
            config = replace(config, max_steps=cfg.steps_per_episode)
        env = build_env(cfg, config)
        obs, _ = env.reset(seed=rng.randrange(1_000_000))
        encoded = encode_observation(obs, device=device)
        state = encoded.state
        head_mask = encoded.head_mask
        target_mask = encoded.target_mask
        color_count = encoded.color_count
        mask = mask_to_tensor(obs["action_mask"], device=device)

        episode_reward = 0.0
        steps = 0
        final_info: dict[str, Any] | None = None
        terminated_flag = False
        truncated_flag = False
        frames: list[str] | None = [env.board_string] if record_rollouts else None
        action_trace: list[int] | None = [] if record_rollouts else None
        action_debug_info: list[dict[str, Any]] | None = [] if record_rollouts else None
        episode_breakdown: dict[str, float] = {}

        while True:
            if not torch.any(mask > 0.0):
                truncated_flag = True
                break
            prev_board = env.board_string if record_rollouts else None
            action = select_action(
                policy_net,
                state,
                mask,
                head_mask,
                target_mask,
                color_count,
                epsilon=epsilon,
                rng=rng,
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Accumulate reward breakdown for episode-level tracking
            if info and "reward_breakdown" in info:
                for component, value in info["reward_breakdown"].items():
                    episode_breakdown[component] = episode_breakdown.get(component, 0.0) + value

            if frames is not None:
                frames.append(env.board_string)
            if action_trace is not None:
                action_trace.append(action)
            if action_debug_info is not None:
                # Track detailed action information for debugging
                board_changed = prev_board != env.board_string
                action_debug_info.append({
                    "action": action,
                    "color": action // 5,
                    "direction": ["UP", "RIGHT", "DOWN", "LEFT", "UNDO"][action % 5],
                    "legal": info.get("legal", True),
                    "reason": info.get("reason", None),
                    "constraint_violations": info.get("constraint_violations", []),
                    "board_changed": board_changed,
                    "reward": reward,
                })

            done = terminated or truncated
            next_encoded = encode_observation(next_obs, device=device)
            state = next_encoded.state
            head_mask = next_encoded.head_mask
            target_mask = next_encoded.target_mask
            color_count = next_encoded.color_count
            mask = mask_to_tensor(next_obs["action_mask"], device=device)

            if done:
                final_info = info
                terminated_flag = bool(terminated)
                truncated_flag = bool(truncated)
                if terminated and not truncated:
                    solved += 1
                break

        total_reward += episode_reward

        if details is not None:
            constraint_penalty = final_info.get("constraint_penalty", 0.0) if final_info else 0.0
            violations = ",".join(final_info.get("constraint_violations", [])) if final_info else ""
            details.append(
                {
                    "episode": eval_idx + 1,
                    "board_idx": config.board_idx or "",
                    "board_size": config.width,
                    "color_count": config.color_count,
                    "puzzle": config.puzzle,
                    "steps": steps,
                    "reward": episode_reward,
                    "solved": int(terminated_flag and not truncated_flag),
                    "terminated": int(terminated_flag),
                    "truncated": int(truncated_flag),
                    "constraint_penalty": constraint_penalty,
                    "constraint_violations": violations,
                    "loop_penalty": episode_breakdown.get("loop", 0.0),
                    "progress_bonus": episode_breakdown.get("progress", 0.0),
                }
            )

        if record_rollouts and rollout_callback and frames is not None:
            rollout_callback(
                config,
                frames,
                action_trace or [],
                terminated_flag and not truncated_flag,
                episode_reward,
                episode_breakdown,
                action_debug_info or [],
            )

    policy_net.train()
    return solved / max(1, episodes), total_reward / max(1, episodes), details


def collect_policy_rollout(
    policy_net: FlowQNetwork,
    config: PuzzleConfig,
    *,
    device: torch.device,
    epsilon: float,
    seed: int,
    cfg: DQNTrainingConfig,
) -> tuple[list[str], bool, float]:
    env = build_env(cfg, config)
    obs, _ = env.reset(seed=seed)
    frames = [env.board_string]
    rng = random.Random(seed)
    total_reward = 0.0

    while True:
        encoded = encode_observation(obs, device=device)
        state = encoded.state
        head_mask = encoded.head_mask
        target_mask = encoded.target_mask
        color_count = encoded.color_count
        mask = mask_to_tensor(obs["action_mask"], device=device)
        action = select_action(
            policy_net,
            state,
            mask,
            head_mask,
            target_mask,
            color_count,
            epsilon=epsilon,
            rng=rng,
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.board_string)
        if terminated or truncated:
            solved = bool(terminated and not truncated)
            break

    return frames, solved, total_reward


def save_rollout_frames(
    frames: list[str],
    rollout_dir: Path,
    config: PuzzleConfig,
    *,
    episode: int,
    solved: bool,
    total_reward: float,
    tag: str,
    make_gif: bool,
    gif_duration: int,
    board_idx: str | None = None,
    reward_breakdown: dict[str, float] | None = None,
    action_debug_info: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path | None, Path]:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    status = "solved" if solved else "unsolved"
    base_name = f"{tag}_ep{episode:04d}_{config.width}x{config.height}_{status}_{timestamp}"
    jsonl_path = rollout_dir / f"{base_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        # First frame is initial state (no action taken yet)
        if action_debug_info:
            # Write detailed JSON per frame with action debug info
            handle.write(json.dumps({"board": frames[0], "action": None}) + "\n")
            for i, (frame, debug_info) in enumerate(zip(frames[1:], action_debug_info), start=1):
                entry = {"board": frame, **debug_info}
                handle.write(json.dumps(entry) + "\n")
        else:
            # Legacy format: just board strings
            for frame in frames:
                handle.write(frame + "\n")

    meta_path = rollout_dir / f"{base_name}.meta.json"
    meta_payload = {
        "episode": episode,
        "tag": tag,
        "solved": solved,
        "total_reward": total_reward,
        "board_width": config.width,
        "board_height": config.height,
        "color_count": config.color_count,
        "frame_count": len(frames),
    }
    if board_idx is not None:
        meta_payload["board_idx"] = board_idx
    if reward_breakdown:
        meta_payload["reward_breakdown"] = reward_breakdown
    meta_path.write_text(json.dumps(meta_payload, indent=2))

    gif_path: Path | None = None
    # GIF generation disabled - use scripts/render_holdout_rollouts.py instead
    if make_gif:
        print(f"[rollout] GIF generation disabled in trainer. Use scripts/render_holdout_rollouts.py to render rollouts.")

    return jsonl_path, gif_path, meta_path


def run_training(
    cfg: DQNTrainingConfig,
    *,
    logger: RunLogger | None = None,
    eval_callback: Callable[[int, float, float], None] | None = None,
) -> tuple[Path, float, float | None]:
    """Run DQN training (backward compatibility wrapper).

    This function now uses the refactored DQNTrainer class internally.
    All functionality is preserved while reducing code duplication.

    Args:
        cfg: DQN training configuration
        logger: Optional MLflow logger
        eval_callback: Optional callback(episode, success_rate, avg_reward)

    Returns:
        Tuple of (output_path, best_train_success, best_validation_success)
    """
    from rl.solver.trainers.dqn_compat import run_training_with_new_trainer

    return run_training_with_new_trainer(cfg, logger=logger, eval_callback=eval_callback)


__all__ = [
    'DQNTrainingConfig',  # DEPRECATED - use modular configs instead
    'PuzzleConfig',
    'epsilon_by_step',
    'select_action',
    'compute_td_loss',
    'load_puzzle_configs',
    'run_episode',
    'evaluate_policy',
    'collect_policy_rollout',
    'save_rollout_frames',
    'run_training',  # DEPRECATED - use DQNTrainer class instead
    'build_env',
]
class FlowEnv(Protocol):
    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        ...

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        ...

    @property
    def board_string(self) -> str:
        ...
