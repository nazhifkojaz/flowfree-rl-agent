from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from rl.env.config import BoardShape
from rl.env.constants import ACTIONS_PER_COLOR, DIRS, EMPTY, UNDO_INDEX
from rl.env.utils import string_to_tokens


@dataclass(frozen=True)
class CellChange:
    index: int
    previous: int
    current: int


@dataclass(frozen=True)
class StateDiff:
    color: int
    from_index: int
    to_index: int
    changed: Tuple[CellChange, ...]
    completed_now: bool
    undone: bool = False
    reverted_completion: bool = False

    @property
    def touched_indices(self) -> Tuple[int, ...]:
        return tuple(change.index for change in self.changed)


@dataclass(frozen=True)
class TransitionOutcome:
    next_state: "BoardState"
    diff: StateDiff | None
    legal: bool
    reason: str | None = None


@dataclass(frozen=True)
class BoardState:
    """
    Immutable snapshot of the FlowFree board.

    Cells follow the encoding: -1 empty, otherwise 1..C color IDs.
    Head and target positions are stored per color (0-indexed).
    """

    shape: BoardShape
    cells: Tuple[int, ...]
    head_positions: Tuple[int, ...]
    start_positions: Tuple[int, ...]
    target_positions: Tuple[int, ...]
    completed: Tuple[bool, ...]
    steps: int
    distance_to_target: Tuple[int, ...]
    paths: Tuple[Tuple[int, ...], ...]

    @staticmethod
    def from_puzzle(shape: BoardShape, puzzle: str) -> "BoardState":
        tokens = string_to_tokens(puzzle, shape.width, shape.height, shape.color_count)
        if tokens is None:
            raise ValueError("Failed to parse puzzle string")

        cells = [-1] * shape.cell_count
        head_positions = [0] * shape.color_count
        start_positions = [0] * shape.color_count
        target_positions = [0] * shape.color_count
        completed = [False] * shape.color_count

        seen = [[] for _ in range(shape.color_count)]
        for idx, token in enumerate(tokens):
            if token.lower() == "x":
                cells[idx] = EMPTY
                continue
            color = int(token) - 1
            if color < 0 or color >= shape.color_count:
                raise ValueError(f"Color {color + 1} out of bounds")
            cells[idx] = color + 1
            seen[color].append(idx)

        for color, endpoints in enumerate(seen):
            if len(endpoints) != 2:
                raise ValueError(f"Color {color + 1} must appear exactly twice (got {len(endpoints)})")
            start, target = endpoints
            start_positions[color] = start
            head_positions[color] = start
            target_positions[color] = target

        distance = [
            manhattan(shape.width, head_positions[c], target_positions[c]) for c in range(shape.color_count)
        ]
        paths = [tuple([head_positions[c]]) for c in range(shape.color_count)]

        return BoardState(
            shape=shape,
            cells=tuple(cells),
            head_positions=tuple(head_positions),
            start_positions=tuple(start_positions),
            target_positions=tuple(target_positions),
            completed=tuple(completed),
            steps=0,
            distance_to_target=tuple(distance),
            paths=tuple(paths),
        )

    def cell(self, index: int) -> int:
        return self.cells[index]

    def indices(self) -> Iterable[int]:
        return range(self.shape.cell_count)

    def neighbors(self, index: int) -> Tuple[int, ...]:
        r, c = divmod(index, self.shape.width)
        results: list[int] = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.shape.height and 0 <= nc < self.shape.width:
                results.append(nr * self.shape.width + nc)
        return tuple(results)

    def update(
        self,
        *,
        cells: Sequence[int] | None = None,
        head_positions: Sequence[int] | None = None,
        completed: Sequence[bool] | None = None,
        steps: int | None = None,
        distance_to_target: Sequence[int] | None = None,
        paths: Sequence[Sequence[int]] | None = None,
    ) -> "BoardState":
        return BoardState(
            shape=self.shape,
            cells=tuple(cells) if cells is not None else self.cells,
            head_positions=tuple(head_positions) if head_positions is not None else self.head_positions,
            start_positions=self.start_positions,
            target_positions=self.target_positions,
            completed=tuple(completed) if completed is not None else self.completed,
            steps=self.steps if steps is None else steps,
            distance_to_target=tuple(distance_to_target) if distance_to_target is not None else self.distance_to_target,
            paths=tuple(tuple(path) for path in paths) if paths is not None else self.paths,
        )

    def all_completed(self) -> bool:
        if not all(self.completed):
            return False
        return all(cell != EMPTY for cell in self.cells)

    def to_string(self) -> str:
        chars = []
        for value in self.cells:
            chars.append("x" if value == EMPTY else str(value))
        return "".join(chars)


def apply_action(state: BoardState, action: int) -> TransitionOutcome:
    color = action // ACTIONS_PER_COLOR
    slot = action % ACTIONS_PER_COLOR

    if color < 0 or color >= state.shape.color_count:
        return TransitionOutcome(next_state=state, diff=None, legal=False, reason="color_out_of_bounds")
    path = list(state.paths[color])

    if slot == UNDO_INDEX:
        if len(path) <= 1:
            return TransitionOutcome(next_state=state, diff=None, legal=False, reason="undo_not_available")

        removed_idx = path[-1]
        new_head_idx = path[-2]

        new_cells = list(state.cells)
        changes: list[CellChange] = []
        if removed_idx != state.target_positions[color] and removed_idx != state.start_positions[color]:
            prev_val = state.cells[removed_idx]
            new_cells[removed_idx] = EMPTY
            changes.append(CellChange(index=removed_idx, previous=prev_val, current=EMPTY))

        new_paths = list(state.paths)
        new_paths[color] = tuple(path[:-1])

        new_heads = list(state.head_positions)
        new_heads[color] = new_head_idx

        new_completed = list(state.completed)
        was_completed = new_completed[color]
        if new_completed[color]:
            new_completed[color] = False

        distance = list(state.distance_to_target)
        distance[color] = manhattan(state.shape.width, new_head_idx, state.target_positions[color])

        new_state = state.update(
            cells=new_cells,
            head_positions=new_heads,
            completed=new_completed,
            steps=state.steps + 1,
            distance_to_target=distance,
            paths=new_paths,
        )

        diff = StateDiff(
            color=color,
            from_index=removed_idx,
            to_index=new_head_idx,
            changed=tuple(changes),
            completed_now=False,
            undone=True,
            reverted_completion=was_completed and not new_completed[color],
        )
        return TransitionOutcome(next_state=new_state, diff=diff, legal=True, reason=None)

    if state.completed[color]:
        return TransitionOutcome(next_state=state, diff=None, legal=False, reason="color_already_completed")

    direction = slot
    from_idx = state.head_positions[color]
    dr, dc = DIRS[direction]
    fr, fc = divmod(from_idx, state.shape.width)
    nr, nc = fr + dr, fc + dc
    if not (0 <= nr < state.shape.height and 0 <= nc < state.shape.width):
        return TransitionOutcome(next_state=state, diff=None, legal=False, reason="out_of_bounds")

    to_idx = nr * state.shape.width + nc
    dest_val = state.cells[to_idx]
    color_val = color + 1
    target_idx = state.target_positions[color]

    if dest_val == EMPTY:
        legal = True
        completed_now = False
    elif to_idx == target_idx and dest_val == color_val:
        legal = True
        completed_now = True
    else:
        return TransitionOutcome(next_state=state, diff=None, legal=False, reason="cell_occupied")

    new_cells = list(state.cells)
    changes: list[CellChange] = []
    if dest_val != color_val:
        new_cells[to_idx] = color_val
        changes.append(CellChange(index=to_idx, previous=dest_val, current=color_val))

    new_heads = list(state.head_positions)
    new_heads[color] = to_idx

    new_completed = list(state.completed)
    if completed_now:
        new_completed[color] = True

    new_paths = list(state.paths)
    new_paths[color] = tuple(list(path) + [to_idx])

    distance = list(state.distance_to_target)
    distance[color] = manhattan(state.shape.width, new_heads[color], target_idx)

    new_state = state.update(
        cells=new_cells,
        head_positions=new_heads,
        completed=new_completed,
        steps=state.steps + 1,
        distance_to_target=distance,
        paths=new_paths,
    )

    diff = StateDiff(
        color=color,
        from_index=from_idx,
        to_index=to_idx,
        changed=tuple(changes),
        completed_now=completed_now,
        undone=False,
        reverted_completion=False,
    )

    return TransitionOutcome(next_state=new_state, diff=diff, legal=legal, reason=None)


def manhattan(width: int, a_index: int, b_index: int) -> int:
    ar, ac = divmod(a_index, width)
    br, bc = divmod(b_index, width)
    return abs(ar - br) + abs(ac - bc)


def reachable_empty_region_counts(state: BoardState) -> Tuple[int, ...]:
    """
    Compute number of empty neighbors for each empty cell.
    Useful for congestion / dead-end metrics.
    """

    counts = [0] * state.shape.cell_count
    for idx, val in enumerate(state.cells):
        if val != EMPTY:
            continue
        total = 0
        for nb in state.neighbors(idx):
            if state.cells[nb] == EMPTY:
                total += 1
        counts[idx] = total
    return tuple(counts)


def bfs_reachable(state: BoardState, start: int, allowed: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Breadth-first search limited to the provided `allowed` set of values.
    Returns the visited indices as a tuple (for deterministic caching).
    """

    visited = np.zeros(state.shape.cell_count, dtype=np.bool_)
    queue = [start]
    visited[start] = True

    while queue:
        idx = queue.pop(0)
        for nb in state.neighbors(idx):
            if visited[nb]:
                continue
            if state.cells[nb] in allowed:
                visited[nb] = True
                queue.append(nb)

    return tuple(int(i) for i, flag in enumerate(visited) if flag)


__all__ = [
    "BoardState",
    "StateDiff",
    "CellChange",
    "TransitionOutcome",
    "apply_action",
    "reachable_empty_region_counts",
    "bfs_reachable",
    "DIRS",
    "ACTIONS_PER_COLOR",
    "UNDO_INDEX",
    "EMPTY",
]
