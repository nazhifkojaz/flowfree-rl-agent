"""Grid utility functions for trace generation."""

from __future__ import annotations

import hashlib

from rl.env.constants import ACTIONS_PER_COLOR


def neighbors(idx: int, width: int, height: int) -> list[int]:
    """Return indices of grid cells adjacent to the given index.

    Args:
        idx: Flat index in row-major order
        width: Grid width
        height: Grid height

    Returns:
        List of valid neighbor indices (up to 4)
    """
    r, c = divmod(idx, width)
    out: list[int] = []
    if r > 0:
        out.append(idx - width)
    if r < height - 1:
        out.append(idx + width)
    if c > 0:
        out.append(idx - 1)
    if c < width - 1:
        out.append(idx + 1)
    return out


def encode_move(from_idx: int, to_idx: int, color: int, width: int) -> int:
    """Encode a move from one cell to an adjacent cell as an action index.

    Args:
        from_idx: Source cell index
        to_idx: Destination cell index (must be adjacent)
        color: Color being moved (1-indexed)
        width: Grid width

    Returns:
        Action index for this move

    Raises:
        ValueError: If move is not between adjacent cells
    """
    fr, fc = divmod(from_idx, width)
    tr, tc = divmod(to_idx, width)
    dr, dc = tr - fr, tc - fc

    # Map direction to index: UP=0, RIGHT=1, DOWN=2, LEFT=3
    if (dr, dc) == (-1, 0):
        dir_idx = 0  # UP
    elif (dr, dc) == (0, 1):
        dir_idx = 1  # RIGHT
    elif (dr, dc) == (1, 0):
        dir_idx = 2  # DOWN
    elif (dr, dc) == (0, -1):
        dir_idx = 3  # LEFT
    else:
        raise ValueError("Move is not axis-aligned between adjacent cells")

    return (color - 1) * ACTIONS_PER_COLOR + dir_idx


def hash_puzzle(puzzle: str) -> str:
    """Generate a short hash of a puzzle string for use in filenames.

    Args:
        puzzle: Puzzle string

    Returns:
        12-character hex hash
    """
    return hashlib.sha1(puzzle.encode("utf-8")).hexdigest()[:12]


__all__ = ["neighbors", "encode_move", "hash_puzzle"]
