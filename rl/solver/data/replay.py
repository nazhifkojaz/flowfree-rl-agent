from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Deque, Sequence
from collections import deque


class SumTree:
    """SumTree data structure for efficient prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: object) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


@dataclass
class Transition:
    state: torch.Tensor
    mask: torch.Tensor
    head_mask: torch.Tensor
    target_mask: torch.Tensor
    color_count: int
    action: int
    reward: float
    next_state: torch.Tensor
    next_mask: torch.Tensor
    next_head_mask: torch.Tensor
    next_target_mask: torch.Tensor
    next_color_count: int
    done: bool


class ReplayBuffer:
    """Replay buffer supporting both uniform and prioritized experience replay."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 1e-4):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        head_mask: torch.Tensor,
        target_mask: torch.Tensor,
        color_count: int,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        next_mask: torch.Tensor,
        next_head_mask: torch.Tensor,
        next_target_mask: torch.Tensor,
        next_color_count: int,
        done: bool,
    ) -> Transition:
        transition = Transition(
            state=state.detach().cpu(),
            mask=mask.detach().cpu(),
            head_mask=head_mask.detach().cpu(),
            target_mask=target_mask.detach().cpu(),
            color_count=int(color_count),
            action=int(action),
            reward=float(reward),
            next_state=next_state.detach().cpu(),
            next_mask=next_mask.detach().cpu(),
            next_head_mask=next_head_mask.detach().cpu(),
            next_target_mask=next_target_mask.detach().cpu(),
            next_color_count=int(next_color_count),
            done=bool(done),
        )
        self._buffer.append(transition)
        self.sum_tree.add(self.max_priority, transition)
        return transition

    def push_transition(self, transition: Transition) -> None:
        self._buffer.append(transition)
        self.sum_tree.add(self.max_priority, transition)

    def sample(self, batch_size: int, prioritized: bool = False) -> tuple[Sequence[Transition], np.ndarray | None, np.ndarray | None]:
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more elements than present in buffer.")

        if not prioritized:
            import random
            indices = random.sample(range(len(self._buffer)), batch_size)
            return [self._buffer[idx] for idx in indices], None, None

        indices = []
        priorities = []
        transitions = []
        total_p = self.sum_tree.total()
        if not np.isfinite(total_p) or total_p <= 0.0:
            import random
            rand_indices = random.sample(range(len(self._buffer)), batch_size)
            return [self._buffer[idx] for idx in rand_indices], None, None

        for _ in range(batch_size):
            s = np.random.uniform(0, total_p)
            idx, priority, data = self.sum_tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            transitions.append(data)

        probs = np.array(priorities, dtype=np.float64) / total_p
        probs = np.maximum(probs, self.epsilon)
        weights = (len(self._buffer) * probs) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)
        return transitions, np.array(indices), weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            if not np.isfinite(priority) or priority <= 0.0:
                priority = self.epsilon
            self.max_priority = max(self.max_priority, priority)
            self.sum_tree.update(idx, priority)


__all__ = ["ReplayBuffer", "Transition"]
