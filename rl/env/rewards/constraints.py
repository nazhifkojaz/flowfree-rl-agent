from __future__ import annotations

from dataclasses import dataclass

from rl.env.rewards.base import RewardContext, RewardEngine, RewardResult
from rl.env.state import BoardState, EMPTY


@dataclass
class ConstraintPenalty(RewardEngine):
    dead_pocket_penalty: float
    disconnect_penalty: float
    degree_penalty: float

    def compute(self, ctx: RewardContext) -> RewardResult:
        if ctx.diff is None:
            return RewardResult(value=0.0, breakdown={})

        breakdown: dict[str, float] = {}
        reward = 0.0

        # Dead pocket detection
        affected_colors = _collect_affected_colors(ctx.next_state, ctx.diff)

        if self.dead_pocket_penalty and has_dead_pocket(ctx.next_state):
            reward += self.dead_pocket_penalty
            breakdown["dead_pocket"] = self.dead_pocket_penalty

        if self.disconnect_penalty and has_disconnect(ctx.next_state, colors=affected_colors):
            reward += self.disconnect_penalty
            breakdown["disconnect"] = self.disconnect_penalty

        if self.degree_penalty and has_degree_violation(ctx.next_state, colors=affected_colors):
            reward += self.degree_penalty
            breakdown["degree"] = self.degree_penalty

        return RewardResult(value=reward, breakdown=breakdown)


def _collect_affected_colors(state: BoardState, diff) -> set[int]:
    affected: set[int] = {diff.color}
    for change in diff.changed:
        idx = change.index
        affected.update(_neighbour_colors(state, idx))
    affected = {c for c in affected if 0 <= c < state.shape.color_count}
    return affected


def _neighbour_colors(state: BoardState, idx: int) -> set[int]:
    colors: set[int] = set()
    for nb in state.neighbors(idx):
        val = state.cells[nb]
        if val > 0:
            colors.add(val - 1)
    return colors


def has_dead_pocket(state: BoardState) -> bool:
    height, width = state.shape.height, state.shape.width
    seen = [False] * state.shape.cell_count

    for idx, val in enumerate(state.cells):
        if val != EMPTY or seen[idx]:
            continue
        queue = [idx]
        seen[idx] = True
        ports = 0
        while queue:
            node = queue.pop()
            r, c = divmod(node, width)
            for nb in state.neighbors(node):
                if state.cells[nb] == EMPTY and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
                elif state.cells[nb] != EMPTY:
                    ports += 1
        if ports <= 1:
            return True
    return False


def has_disconnect(state: BoardState, *, colors: set[int] | None = None) -> bool:
    if colors is None:
        color_iterable = range(state.shape.color_count)
    else:
        color_iterable = colors
    for color in color_iterable:
        if color >= state.shape.color_count:
            continue
        completed = state.completed[color]
        if completed:
            continue
        head = state.head_positions[color]
        target = state.target_positions[color]
        stack = [head]
        visited = set([head])
        allowed = {EMPTY, color + 1}

        while stack:
            node = stack.pop()
            if node == target:
                break
            for nb in state.neighbors(node):
                if nb in visited:
                    continue
                if state.cells[nb] in allowed:
                    visited.add(nb)
                    stack.append(nb)
        else:
            return True
    return False


def has_degree_violation(state: BoardState, *, colors: set[int] | None = None) -> bool:
    start_positions = set(state.start_positions)
    target_positions = set(state.target_positions)
    if colors is None:
        color_iterable = range(state.shape.color_count)
    else:
        color_iterable = colors
    for color in color_iterable:
        if color >= state.shape.color_count:
            continue
        color_val = color + 1
        for idx, val in enumerate(state.cells):
            if val != color_val:
                continue
            same_neighbors = sum(1 for nb in state.neighbors(idx) if state.cells[nb] == color_val)
            is_endpoint = idx in start_positions or idx in target_positions
            limit = 1 if is_endpoint else 2
            if same_neighbors > limit:
                return True
    return False
