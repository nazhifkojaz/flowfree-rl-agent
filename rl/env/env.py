from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Tuple

import numpy as np

from rl.env.config import EnvConfig
from rl.env.mask import ActionMasker
from rl.env.observation import ObservationBuilder
from rl.env.rewards import RewardSystem, build_reward_system
from rl.env.rewards.base import RewardContext
from rl.env.state import BoardState, TransitionOutcome, apply_action
from rl.env.constants import EMPTY, ACTIONS_PER_COLOR


class FlowFreeEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self._reward_system: RewardSystem = build_reward_system(config.reward)
        self._masker = ActionMasker(config.mask)
        self._obs_builder = ObservationBuilder(config.observation)
        self._rng = np.random.default_rng(config.seed)

        self._state: BoardState | None = None
        self._mask: np.ndarray | None = None
        self._last_action: int | None = None
        self._max_steps = config.effective_max_steps
        params = config.reward.params
        self._loop_penalty: float = params.get("loop_penalty", 0.0)
        self._loop_window: int = int(params.get("loop_window", 0) or 0)
        self._progress_bonus: float = params.get("progress_bonus", 0.0)
        history_len = max(3, self._loop_window) if self._loop_penalty else 0
        self._head_histories: list[Deque[int]] | None = (
            [deque(maxlen=history_len) for _ in range(self.config.shape.color_count)]
            if history_len > 0
            else None
        )
        self._last_fill_count: int = 0
        # Track action history for undo-redo oscillation detection
        self._action_history: Deque[int] = deque(maxlen=3)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None) -> Tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = BoardState.from_puzzle(self.config.shape, self.config.puzzle)
        self._mask = self._masker.initial_mask(self._state)
        self._last_action = None
        self._action_history.clear()
        if self._head_histories is not None:
            for color, history in enumerate(self._head_histories):
                history.clear()
                history.append(self._state.head_positions[color])
        self._last_fill_count = sum(1 for cell in self._state.cells if cell != EMPTY)

        observation = self._obs_builder.build(
            self._state,
            self._mask,
            last_action=self._last_action,
            max_steps=self._max_steps,
        )
        info = {"action_mask": observation["action_mask"].copy(), "reward_breakdown": {}}
        return observation, info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        if self._state is None or self._mask is None:
            raise RuntimeError("Environment not initialised. Call reset() before step().")

        outcome = apply_action(self._state, action)

        if not outcome.legal:
            # Illegal action: apply penalty and increment step counter
            penalty = self._reward_system.invalid_penalty
            next_state = self._state.update(steps=self._state.steps + 1)
            reward = penalty
            breakdown = {"invalid": penalty} if penalty else {}
            terminated = False
            truncated = next_state.steps >= self.config.effective_max_steps
            diff = None
        else:
            next_state = outcome.next_state
            terminated = next_state.all_completed()
            truncated = next_state.steps >= self.config.effective_max_steps and not terminated

            ctx = RewardContext(
                previous=self._state,
                next_state=next_state,
                diff=outcome.diff,
                terminated=terminated,
                truncated=truncated,
            )
            reward_result = self._reward_system.engine.compute(ctx)
            reward = reward_result.value
            breakdown = dict(reward_result.breakdown)

            if outcome.diff and outcome.diff.undone and self._reward_system.undo_penalty:
                reward += self._reward_system.undo_penalty
                breakdown["undo"] = breakdown.get("undo", 0.0) + self._reward_system.undo_penalty

            # Apply unsolved penalty if truncated without solving
            if truncated and self._reward_system.unsolved_penalty and not terminated:
                reward += self._reward_system.unsolved_penalty
                breakdown["unsolved"] = breakdown.get("unsolved", 0.0) + self._reward_system.unsolved_penalty

            # Apply efficiency bonus if solved (rewards solving with fewer steps)
            if terminated and self._reward_system.solve_efficiency_bonus:
                max_steps = self.config.effective_max_steps
                steps_remaining = max(0, max_steps - next_state.steps)
                efficiency_reward = steps_remaining * self._reward_system.solve_efficiency_bonus
                reward += efficiency_reward
                breakdown["solve_efficiency"] = efficiency_reward

            diff = outcome.diff

        self._state = next_state
        self._mask = self._masker.update(self._state, self._mask, diff)
        self._last_action = action if outcome.legal else None

        # If no valid actions remain and puzzle not solved, force truncation
        if self._mask.sum() == 0 and not terminated:
            truncated = True
            # Apply unsolved penalty since we're forced to truncate
            if self._reward_system.unsolved_penalty:
                reward += self._reward_system.unsolved_penalty
                breakdown["unsolved"] = breakdown.get("unsolved", 0.0) + self._reward_system.unsolved_penalty

        observation = self._obs_builder.build(
            self._state,
            self._mask,
            last_action=self._last_action,
            max_steps=self._max_steps,
        )
        constraint_keys = {"dead_pocket", "disconnect", "degree"}
        constraint_penalty = sum(breakdown.get(key, 0.0) for key in constraint_keys)
        constraint_flags = [key for key in constraint_keys if breakdown.get(key)]

        fill_count = sum(1 for cell in self._state.cells if cell != EMPTY)
        if self._progress_bonus:
            delta = max(0, fill_count - self._last_fill_count)
            if delta > 0:
                bonus = self._progress_bonus * delta
                reward += bonus
                breakdown["progress"] = breakdown.get("progress", 0.0) + bonus
        self._last_fill_count = fill_count

        if outcome.legal and self._head_histories is not None and self._loop_penalty:
            color_idx = action // ACTIONS_PER_COLOR
            history = self._head_histories[color_idx]
            history.append(self._state.head_positions[color_idx])
            if len(history) >= 3:
                if history[-1] == history[-3] and history[-1] != history[-2]:
                    reward += self._loop_penalty
                    breakdown["loop"] = breakdown.get("loop", 0.0) + self._loop_penalty

        # Detect undo-redo oscillation: if we just undid and now redoing same move
        if outcome.legal and len(self._action_history) >= 2:
            prev_action = self._action_history[-1]
            prev_prev_action = self._action_history[-2]
            current_color = action // ACTIONS_PER_COLOR
            current_slot = action % ACTIONS_PER_COLOR
            prev_color = prev_action // ACTIONS_PER_COLOR
            prev_slot = prev_action % ACTIONS_PER_COLOR
            prev_prev_color = prev_prev_action // ACTIONS_PER_COLOR

            # Detect: [move] → [undo same color] → [same move] pattern
            if (current_color == prev_color == prev_prev_color and
                prev_slot == 4 and  # Previous was undo (UNDO_INDEX = 4)
                current_slot == (prev_prev_action % ACTIONS_PER_COLOR) and  # Same move as before undo
                current_slot != 4):  # Current is not undo
                # Heavy penalty for undo-redo oscillation
                oscillation_penalty = self._loop_penalty * 5.0  # 5x stronger than normal loop penalty
                reward += oscillation_penalty
                breakdown["undo_redo_oscillation"] = breakdown.get("undo_redo_oscillation", 0.0) + oscillation_penalty

        # Track action history
        if outcome.legal:
            self._action_history.append(action)
            
        info: Dict[str, Any] = {
            "legal": outcome.legal,
            "reason": outcome.reason,
            "reward_breakdown": breakdown,
            "action_mask": observation["action_mask"].copy(),
            "constraint_penalty": constraint_penalty,
            "constraint_violations": constraint_flags,
            "undone": bool(diff.undone) if diff is not None else False,
            "reopened_color": diff.color if diff is not None and diff.reverted_completion else None,
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> str:
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._state.to_string()

    @property
    def board_string(self) -> str:
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._state.to_string()

    @property
    def action_mask(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._mask.copy()
