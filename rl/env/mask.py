from __future__ import annotations

import numpy as np

from rl.env.config import MaskConfig
from rl.env.state import ACTIONS_PER_COLOR, UNDO_INDEX, BoardState, DIRS, EMPTY


class ActionMasker:
    def __init__(self, config: MaskConfig):
        self.config = config

    def initial_mask(self, state: BoardState) -> np.ndarray:
        return compute_full_mask(state)

    def update(self, state: BoardState, mask: np.ndarray, diff) -> np.ndarray:
        return compute_full_mask(state)

    def _neighboring_colors(self, state: BoardState, diff) -> set[int]:
        touched = set(diff.touched_indices)
        colors: set[int] = set()
        for color, head in enumerate(state.head_positions):
            if color == diff.color or state.completed[color]:
                continue
            if head in touched:
                colors.add(color)
        return colors

    def _write_color_mask(self, state: BoardState, color: int, mask: np.ndarray) -> None:
        start = color * ACTIONS_PER_COLOR
        end = start + ACTIONS_PER_COLOR
        mask[start:end] = 0

        path_len = len(state.paths[color])
        # Allow undo if path has more than just the starting position
        if path_len > 1:
            mask[start + UNDO_INDEX] = 1

        # If completed, allow undo but lock all directional moves
        if state.completed[color]:
            return

        head = state.head_positions[color]
        hr, hc = divmod(head, state.shape.width)
        target = state.target_positions[color]
        color_val = color + 1

        for dir_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = hr + dr, hc + dc
            if not (0 <= nr < state.shape.height and 0 <= nc < state.shape.width):
                continue
            dest = nr * state.shape.width + nc
            val = state.cells[dest]
            if val == EMPTY or (dest == target and val == color_val):
                mask[start + dir_idx] = 1


def compute_full_mask(state: BoardState) -> np.ndarray:
    mask = np.zeros(state.shape.action_dim, dtype=np.int8)
    for color in range(state.shape.color_count):
        start = color * ACTIONS_PER_COLOR
        path_len = len(state.paths[color])
        if path_len > 1:
            mask[start + UNDO_INDEX] = 1

        # Completed colours should only allow undo (if available)
        if state.completed[color]:
            continue
        head = state.head_positions[color]
        hr, hc = divmod(head, state.shape.width)
        target = state.target_positions[color]
        color_val = color + 1
        for dir_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = hr + dr, hc + dc
            if not (0 <= nr < state.shape.height and 0 <= nc < state.shape.width):
                continue
            dest = nr * state.shape.width + nc
            val = state.cells[dest]
            if val == EMPTY or (dest == target and val == color_val):
                mask[start + dir_idx] = 1

    # Safety check: all actions masked but puzzle not solved
    # This should never happen - env.py forces truncation in this case
    # Kept for debugging edge cases during development
    if mask.sum() == 0 and not state.all_completed():
        pass  # Environment handles this gracefully

    return mask


__all__ = ["ActionMasker", "compute_full_mask"]
