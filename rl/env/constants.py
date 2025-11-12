from __future__ import annotations

from typing import Tuple

EMPTY = -1

# Cardinal directions (row, col)
DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTIONS_PER_COLOR = len(DIRS) + 1  # 4 directional moves + 1 undo
UNDO_INDEX = ACTIONS_PER_COLOR - 1

__all__ = ["EMPTY", "DIRS", "ACTIONS_PER_COLOR", "UNDO_INDEX"]
