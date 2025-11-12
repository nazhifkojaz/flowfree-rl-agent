from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from rl.env.env import FlowFreeEnv
from rl.env.config import BoardShape, EnvConfig, DEFAULT_OBSERVATION, MaskConfig, RewardPreset
from rl.env.config import BoardShape, EnvConfig, DEFAULT_OBSERVATION, MaskConfig, ObservationSpec, RewardPreset, default_max_steps
from rl.solver.reward_settings import RewardSettings
from rl.solver.constants import MAX_CHANNELS
from rl.solver.data import ReplayBuffer, Transition
from rl.solver.observation import EncodedObservation, encode_observation, mask_to_tensor
from rl.solver.policies.q_network import FlowQNetwork
from rl.solver.policies.policy import FlowPolicy, load_policy
from rl.solver.core import NullRunLogger, RunLogger, RunPaths


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
    complete_sustain_bonus: float = 0.0
    complete_revert_penalty: float | None = None
    solve_bonus: float = 20.0
    constraint_free_bonus: float = 5.0
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
    env_backend: str = "legacy"
    env2_reward: str = "potential"
    env2_channels: tuple[str, ...] | None = None
    env2_undo_penalty: float = -0.1
    expert_buffer_size: int = 0
    expert_sample_ratio: float = 0.0
    loop_penalty: float = 0.0
    loop_window: int = 0
    progress_bonus: float = 0.0


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
) -> tuple[float, int, bool, list[float], int, float, Counter[str], dict[str, float]]:
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
            optimizer.step()
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

    return total_reward, steps, solved, losses, global_step + steps, constraint_penalty_total, constraint_counts, reward_breakdown


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


def ensure_log_dir(log_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = log_root / f"dqn_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_hyperparams(log_dir: Path, config: DQNTrainingConfig, puzzle_count: int) -> None:
    payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }
    payload["puzzle_count"] = puzzle_count
    path = log_dir / "hyperparams.json"
    path.write_text(json.dumps(payload, indent=2))


def run_training(
    cfg: DQNTrainingConfig,
    *,
    logger: RunLogger | None = None,
    eval_callback: Callable[[int, float, float], None] | None = None,
) -> tuple[Path, float, float | None]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = logger or NullRunLogger()

    # Use simplified rewards if flag is set
    if cfg.simple_rewards:
        from rl.solver.reward_settings import get_simple_reward_settings

        reward_cfg = get_simple_reward_settings()
        print("Using simplified reward configuration (potential-based only)")
    else:
        reward_cfg = RewardSettings(
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
    configs = load_puzzle_configs(
        cfg.puzzle_csv,
        limit=cfg.puzzle_limit,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        max_colors=cfg.max_colors,
        reward_cfg=reward_cfg,
        seed=cfg.seed,
    )
    if not configs:
        logger.close()
        raise SystemExit("No puzzles available for training. Check CSV filters.")

    params = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(cfg).items()
    }
    params["puzzle_count"] = len(configs)
    validation_configs: list[PuzzleConfig] = []
    if cfg.validation_csv is not None:
        validation_configs = load_puzzle_configs(
            cfg.validation_csv,
            limit=cfg.validation_limit,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            max_colors=cfg.max_colors,
            reward_cfg=reward_cfg,
            seed=cfg.seed + 131,
        )
        params["validation_count"] = len(validation_configs)
    logger.log_params(params)
    logger.set_tags({"phase": "dqn_finetune"})

    device = torch.device(cfg.device)
    policy_net = FlowQNetwork(in_channels=MAX_CHANNELS, use_dueling=cfg.use_dueling).to(device)
    if cfg.policy_init is not None:
        if not cfg.policy_init.exists():
            raise SystemExit(f"Policy init checkpoint not found: {cfg.policy_init}")
        supervised_policy = FlowPolicy(in_channels=MAX_CHANNELS)
        load_policy(supervised_policy, cfg.policy_init, map_location=device)
        policy_net.backbone.load_state_dict(supervised_policy.backbone.state_dict(), strict=False)
        print(f"Loaded supervised backbone weights from {cfg.policy_init}")
    target_net = FlowQNetwork(in_channels=MAX_CHANNELS, use_dueling=cfg.use_dueling).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha, beta=cfg.per_beta, beta_increment=cfg.per_beta_increment)
    expert_buffer: ReplayBuffer | None = None
    if cfg.expert_buffer_size > 0 and cfg.expert_sample_ratio > 0:
        expert_buffer = ReplayBuffer(cfg.expert_buffer_size, alpha=0.0, beta=0.0, beta_increment=0.0)

    if cfg.log_dir is not None:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = ensure_log_dir(cfg.log_root)
    write_hyperparams(log_dir, cfg, puzzle_count=len(configs))

    # Initialize TensorBoard writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        print(f"TensorBoard logging enabled: {log_dir / 'tensorboard'}")
    except ImportError:
        tb_writer = None
        print("TensorBoard not available (torch.utils.tensorboard not found)")
    metrics_path = log_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "episode,total_reward,steps,epsilon,avg_loss,solved,constraint_penalty,constraint_dead_pocket,"
            "constraint_disconnect,constraint_degree,eval_success,eval_reward,val_success,val_reward,board_size,"
            "color_count,reward_breakdown\n"
        )
    rollout_root: Path | None = None
    rollouts_recorded = 0
    if cfg.record_rollouts:
        rollout_root = cfg.rollout_dir / log_dir.name
        rollout_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("episode,rolling_reward_100,rolling_success_100,rolling_constraint_penalty_100\n")
    reward_window: deque[float] = deque(maxlen=100)
    success_window: deque[int] = deque(maxlen=100)
    constraint_window: deque[float] = deque(maxlen=100)

    size_buckets: dict[int, list[PuzzleConfig]] = defaultdict(list)
    color_buckets: dict[int, list[PuzzleConfig]] = defaultdict(list)
    for cfg_item in configs:
        size_buckets[cfg_item.width].append(cfg_item)
        color_buckets[cfg_item.color_count].append(cfg_item)
    color_episode_counts: Counter[int] = Counter()
    color_success_counts: Counter[int] = Counter()
    puzzle_episode_counts: Counter[str] = Counter()
    puzzle_success_counts: Counter[str] = Counter()

    unsolved_start = (
        cfg.unsolved_penalty_start
        if cfg.unsolved_penalty_start is not None
        else cfg.unsolved_penalty
    )
    warmup_total = max(0, cfg.unsolved_penalty_warmup)

    def current_unsolved_penalty(ep: int) -> float:
        if warmup_total == 0 or cfg.unsolved_penalty == unsolved_start:
            return cfg.unsolved_penalty
        progress = min(1.0, max(0.0, ep / warmup_total))
        return unsolved_start + progress * (cfg.unsolved_penalty - unsolved_start)

    def current_penalty_scale(ep: int) -> float:
        if cfg.penalty_warmup <= 0:
            return 1.0
        return min(1.0, max(0.0, ep / cfg.penalty_warmup))

    def puzzle_weight(cfg: PuzzleConfig) -> float:
        if not cfg.board_idx:
            return 1.0
        total = puzzle_episode_counts[cfg.board_idx]
        if total == 0:
            return 1.0
        success = puzzle_success_counts[cfg.board_idx]
        rate = success / total
        return max(0.05, 1.0 - rate)

    def choose_from_bucket(bucket: list[PuzzleConfig]) -> PuzzleConfig:
        if not bucket:
            return random.choice(configs)
        weights = [puzzle_weight(cfg) for cfg in bucket]
        return random.choices(bucket, weights=weights, k=1)[0]

    def select_curriculum_config(ep: int) -> PuzzleConfig:
        available_colors = [c for c, bucket in color_buckets.items() if bucket]
        if available_colors:
            weights: list[float] = []
            for color in available_colors:
                total = color_episode_counts[color]
                success = color_success_counts[color]
                if total == 0:
                    weight = 1.0
                else:
                    success_rate = success / total
                    weight = max(0.05, 1.0 - success_rate)
                weights.append(weight)
            chosen_color = random.choices(available_colors, weights=weights, k=1)[0]
            return choose_from_bucket(color_buckets[chosen_color])

        bucket5 = size_buckets.get(5, [])
        bucket6 = size_buckets.get(6, [])
        if bucket5 and bucket6:
            duration = max(0, cfg.curriculum_six_prob_episodes)
            if duration == 0:
                prob_six = cfg.curriculum_six_prob_end
            else:
                progress = min(1.0, max(0.0, ep / duration))
                prob_six = cfg.curriculum_six_prob_start + progress * (
                    cfg.curriculum_six_prob_end - cfg.curriculum_six_prob_start
                )
            prob_six = max(0.0, min(1.0, prob_six))
            if random.random() < prob_six:
                return choose_from_bucket(bucket6)
            return choose_from_bucket(bucket5)
        return choose_from_bucket(configs)

    global_step = 0
    best_train = 0.0
    best_validation: float | None = None
    best_primary = -1.0

    try:
        for episode in range(1, cfg.episodes + 1):
            selected_config = select_curriculum_config(episode)
            config = selected_config
            # Override step cap if requested; otherwise let config/defaults apply (area + DEFAULT_STEP_BUFFER)
            if cfg.steps_per_episode is not None:
                config = replace(config, max_steps=cfg.steps_per_episode)

            penalty = current_unsolved_penalty(episode)
            penalty_scale = current_penalty_scale(episode)
            env_reward = replace(
                config.reward,
                unsolved_penalty=penalty,
                disconnect_penalty=cfg.disconnect_penalty * penalty_scale,
                degree_penalty=cfg.degree_penalty * penalty_scale,
                complete_color_bonus=cfg.complete_color_bonus,
                solve_bonus=cfg.solve_bonus,
            )
            config = replace(config, reward=env_reward)
            env = build_env(cfg, config)
            schedule = (cfg.epsilon_schedule or "linear").lower()
            if schedule == "linear":
                epsilon = epsilon_by_step(
                    episode,
                    start=cfg.epsilon_start,
                    end=cfg.epsilon_end,
                    decay=cfg.epsilon_decay,
                    schedule="linear",
                    linear_total=cfg.epsilon_linear_steps or cfg.episodes,
                )
            else:
                epsilon = epsilon_by_step(
                    global_step,
                    start=cfg.epsilon_start,
                    end=cfg.epsilon_end,
                    decay=cfg.epsilon_decay,
                    schedule="exp",
                )
            record_frames: list[str] | None = None
            capture_training_rollout = (
                cfg.record_rollouts
                and cfg.rollout_frequency > 0
                and (cfg.rollout_max is None or rollouts_recorded < cfg.rollout_max)
                and episode % cfg.rollout_frequency == 0
            )
            if capture_training_rollout:
                record_frames = []

            (
                ep_reward,
                ep_steps,
                solved,
                losses,
                global_step,
                constraint_penalty,
                constraint_counts,
                reward_breakdown,
            ) = run_episode(
                env,
                policy_net,
                cfg=cfg,
                epsilon=epsilon,
                device=device,
                rng=random,
                buffer=buffer,
                optimizer=optimizer,
                target_net=target_net,
                gamma=cfg.gamma,
                batch_size=cfg.batch_size,
                grad_clip=cfg.grad_clip,
                global_step=global_step,
                reward_scale=cfg.reward_scale,
                reward_clamp=cfg.reward_clamp,
                record_frames=record_frames,
                expert_buffer=expert_buffer,
            )

            color_episode_counts[selected_config.color_count] += 1
            if solved:
                color_success_counts[selected_config.color_count] += 1
            if selected_config.board_idx:
                puzzle_episode_counts[selected_config.board_idx] += 1
                if solved:
                    puzzle_success_counts[selected_config.board_idx] += 1

            if global_step % max(1, cfg.target_update) == 0:
                target_net.load_state_dict(policy_net.state_dict())

            avg_loss = sum(losses) / len(losses) if losses else 0.0
            dead_hits = constraint_counts.get("dead_pocket", 0)
            disconnect_hits = constraint_counts.get("disconnect", 0)
            degree_hits = constraint_counts.get("degree", 0)
            eval_success = ""
            eval_reward = ""
            val_success = ""
            val_reward = ""
            val_success_value: float | None = None
            val_avg_reward: float | None = None

            target_steps = config.max_steps or default_max_steps(config.width, config.height)
            high_reward_unsolved = (
                not solved
                and target_steps > 0
                and ep_steps >= 0.9 * target_steps
                and ep_reward >= max(cfg.solve_bonus * 0.5, 5.0)
            )
            if high_reward_unsolved:
                logger.log_metric("warn/high_reward_unsolved", float(ep_reward), episode)
                if tb_writer is not None:
                    tb_writer.add_scalar("warn/high_reward_unsolved", ep_reward, episode)
                print(
                    f"[warn] episode {episode} reward={ep_reward:.3f} steps={ep_steps} solved={solved} "
                    f"(near max {target_steps})"
                )

            reward_window.append(ep_reward)
            success_window.append(1 if solved else 0)
            constraint_window.append(constraint_penalty)
            rolling_reward = sum(reward_window) / len(reward_window)
            rolling_success = sum(success_window) / len(success_window)
            rolling_constraint = sum(constraint_window) / len(constraint_window)
            with summary_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(f"{episode},{rolling_reward:.4f},{rolling_success:.4f},{rolling_constraint:.4f}\n")
            if (
                record_frames is not None
                and rollout_root is not None
                and (solved or cfg.rollout_include_unsolved)
                and (cfg.rollout_max is None or rollouts_recorded < cfg.rollout_max)
            ):
                json_path, gif_path, meta_path = save_rollout_frames(
                    record_frames,
                    rollout_root,
                    config,
                    episode=episode,
                    solved=solved,
                    total_reward=ep_reward,
                    tag="train",
                    make_gif=cfg.rollout_make_gif,
                    gif_duration=cfg.rollout_gif_duration,
                    board_idx=config.board_idx,
                    reward_breakdown=reward_breakdown,
                )
                rollouts_recorded += 1
                print(f"[rollout] Saved training episode {episode} to {json_path}")
                logger.log_artifact(json_path)
                logger.log_artifact(meta_path)
                if gif_path is not None:
                    logger.log_artifact(gif_path)
            if cfg.eval_interval > 0 and episode % cfg.eval_interval == 0:
                success_rate, avg_eval_reward, _ = evaluate_policy(
                    policy_net,
                    configs,
                    device=device,
                    episodes=cfg.eval_episodes,
                    seed=cfg.seed + episode,
                    epsilon=cfg.eval_epsilon,
                    cfg=cfg,
                    record_details=False,
                )
                eval_success = f"{success_rate:.3f}"
                eval_reward = f"{avg_eval_reward:.3f}"
                best_train = max(best_train, success_rate)
                logger.log_metric("eval/success_rate", success_rate, episode)
                logger.log_metric("eval/avg_reward", avg_eval_reward, episode)
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/success_rate", success_rate, episode)
                    tb_writer.add_scalar("eval/avg_reward", avg_eval_reward, episode)
                primary_metric = success_rate
                primary_reward = avg_eval_reward
                if validation_configs:
                    val_episodes = cfg.validation_eval_episodes or len(validation_configs)
                    if val_episodes > 0:
                        val_success_value, val_avg_reward, _ = evaluate_policy(
                            policy_net,
                            validation_configs,
                            device=device,
                            episodes=val_episodes,
                            seed=cfg.seed + episode + 10_000,
                            epsilon=cfg.eval_epsilon,
                            cfg=cfg,
                            record_details=False,
                        )
                        val_success = f"{val_success_value:.3f}"
                        val_reward = f"{val_avg_reward:.3f}"
                        if best_validation is None or val_success_value > best_validation:
                            best_validation = val_success_value
                        logger.log_metric("val/success_rate", val_success_value, episode)
                        logger.log_metric("val/avg_reward", val_avg_reward, episode)
                        if tb_writer is not None:
                            tb_writer.add_scalar("val/success_rate", val_success_value, episode)
                            tb_writer.add_scalar("val/avg_reward", val_avg_reward, episode)
                        primary_metric = val_success_value
                        primary_reward = val_avg_reward
                if primary_metric > best_primary:
                    best_primary = primary_metric
                    torch.save(policy_net.state_dict(), log_dir / "best.pt")
                    if (
                        cfg.record_rollouts
                        and rollout_root is not None
                        and (cfg.rollout_max is None or rollouts_recorded < cfg.rollout_max)
                    ):
                        eval_pool = validation_configs if val_success_value is not None and validation_configs else configs
                        eval_config = random.choice(eval_pool)
                        frames, eval_solved, eval_total_reward = collect_policy_rollout(
                            policy_net,
                            eval_config,
                            device=device,
                            epsilon=cfg.eval_epsilon,
                            seed=cfg.seed + episode + 1,
                            cfg=cfg,
                        )
                        if eval_solved or cfg.rollout_include_unsolved:
                            json_path, gif_path, meta_path = save_rollout_frames(
                                frames,
                                rollout_root,
                                eval_config,
                                episode=episode,
                                solved=eval_solved,
                                total_reward=eval_total_reward,
                                tag="eval",
                                make_gif=cfg.rollout_make_gif,
                                gif_duration=cfg.rollout_gif_duration,
                                board_idx=eval_config.board_idx,
                                reward_breakdown=None,
                            )
                            rollouts_recorded += 1
                            print(f"[rollout] Saved evaluation snapshot at episode {episode} to {json_path}")
                            logger.log_artifact(json_path)
                            logger.log_artifact(meta_path)
                            if gif_path is not None:
                                logger.log_artifact(gif_path)
                msg = (
                    f"[eval] episode={episode} train_success={success_rate:.3f} "
                    f"train_reward={avg_eval_reward:.3f}"
                )
                if val_success_value is not None and val_avg_reward is not None:
                    msg += f" val_success={val_success_value:.3f} val_reward={val_avg_reward:.3f}"
                msg += f" buffer={len(buffer)}"
                print(msg)
                if eval_callback is not None:
                    eval_callback(episode, success_rate, avg_eval_reward)

            reward_json = json.dumps(reward_breakdown, sort_keys=True)
            reward_json_escaped = reward_json.replace('"', '""')
            with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(
                    f"{episode},{ep_reward:.4f},{ep_steps},{epsilon:.4f},{avg_loss:.6f},{int(solved)},"
                    f"{constraint_penalty:.4f},{dead_hits},{disconnect_hits},{degree_hits},{eval_success},{eval_reward},"
                    f"{val_success},{val_reward},{config.width},{config.color_count},\"{reward_json_escaped}\"\n"
                )

            # MLflow logging
            logger.log_metric("train/total_reward", ep_reward, episode)
            logger.log_metric("train/avg_loss", avg_loss, episode)
            logger.log_metric("train/epsilon", epsilon, episode)
            logger.log_metric("train/solved", float(int(solved)), episode)
            logger.log_metric("train/constraint_penalty", constraint_penalty, episode)
            logger.log_metric("train/buffer_size", float(len(buffer)), episode)

            # Log reward breakdown components
            if reward_breakdown:
                for component, total_value in reward_breakdown.items():
                    logger.log_metric(f"reward_components/{component}", total_value, episode)

            # TensorBoard logging (if available)
            if tb_writer is not None:
                tb_writer.add_scalar("train/total_reward", ep_reward, episode)
                tb_writer.add_scalar("train/avg_loss", avg_loss, episode)
                tb_writer.add_scalar("train/epsilon", epsilon, episode)
                tb_writer.add_scalar("train/solved", float(int(solved)), episode)
                tb_writer.add_scalar("train/constraint_penalty", constraint_penalty, episode)
                tb_writer.add_scalar("train/buffer_size", float(len(buffer)), episode)
                tb_writer.add_scalar("train/steps", ep_steps, episode)
                tb_writer.add_scalar("train/constraint_penalty", constraint_penalty, episode)

                # Log reward breakdown to TensorBoard
                if reward_breakdown:
                    for component, total_value in reward_breakdown.items():
                        tb_writer.add_scalar(f"reward_components/{component}", total_value, episode)

            if cfg.log_every and episode % cfg.log_every == 0:
                print(
                    f"[train] episode={episode}/{cfg.episodes} reward={ep_reward:.3f} "
                    f"steps={ep_steps} epsilon={epsilon:.3f} avg_loss={avg_loss:.4f} "
                    f"solved={int(solved)} buffer={len(buffer)}"
                )

        torch.save(policy_net.state_dict(), cfg.output)
        primary_best = best_validation if best_validation is not None else best_train
        logger.log_metric("final/best_success_rate", primary_best, cfg.episodes)
        logger.log_metric("final/best_train_success", best_train, cfg.episodes)
        if best_validation is not None:
            logger.log_metric("final/best_validation_success", best_validation, cfg.episodes)
        logger.log_metric("final/episodes", float(cfg.episodes), cfg.episodes)
        logger.log_metric("final/buffer_size", float(len(buffer)), cfg.episodes)
        logger.log_artifact(metrics_path)
        if (log_dir / "best.pt").exists():
            logger.log_artifact(log_dir / "best.pt")
        if Path(cfg.output).exists():
            logger.log_artifact(cfg.output)

        best_eval_score = best_primary

    finally:
        logger.close()
        if tb_writer is not None:
            tb_writer.close()

    return cfg.output, best_train, best_validation

__all__ = [
    'DQNTrainingConfig',
    'epsilon_by_step',
    'select_action',
    'compute_td_loss',
    'load_puzzle_configs',
    'run_episode',
    'evaluate_policy',
    'collect_policy_rollout',
    'save_rollout_frames',
    'ensure_log_dir',
    'write_hyperparams',
    'run_training',
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
