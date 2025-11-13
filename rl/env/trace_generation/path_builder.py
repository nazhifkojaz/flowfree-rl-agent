"""Path reconstruction and solution validation for trace generation."""

from __future__ import annotations

from rl.env.trace_generation.grid_utils import neighbors


def extract_endpoints(
    puzzle: list[str], width: int, height: int, color_count: int
) -> dict[int, tuple[int, int]]:
    """Extract color endpoint pairs from a puzzle.

    Args:
        puzzle: List of token strings (row-major order)
        width: Grid width
        height: Grid height
        color_count: Number of colors in the puzzle

    Returns:
        Dict mapping color (1-indexed) to (start_idx, end_idx) tuple
    """
    endpoint_map: dict[int, list[int]] = {c: [] for c in range(1, color_count + 1)}
    for idx, tok in enumerate(puzzle):
        if tok.lower() == "x":
            continue
        endpoint_map[int(tok)].append(idx)

    endpoints: dict[int, tuple[int, int]] = {}
    for color, locs in endpoint_map.items():
        if len(locs) == 2:
            endpoints[color] = (locs[0], locs[1])
    return endpoints


def build_color_path(
    color: int,
    endpoints: tuple[int, int],
    final_values: list[int],
    width: int,
    height: int,
) -> list[int]:
    """Reconstruct the path for a single color in a solved puzzle.

    Uses DFS to trace the path from start to goal endpoint, ensuring all
    cells of this color are visited exactly once.

    Args:
        color: Color to trace (1-indexed)
        endpoints: (start_idx, goal_idx) tuple for this color
        final_values: Solved board values (flat list)
        width: Grid width
        height: Grid height

    Returns:
        List of cell indices forming the path from start to goal

    Raises:
        ValueError: If path cannot be reconstructed or doesn't cover all cells
    """
    start_idx, goal_idx = endpoints
    color_cells = [i for i, v in enumerate(final_values) if v == color]
    if not color_cells:
        return []

    path: list[int] = []

    def dfs(current: int, prev: int | None) -> bool:
        path.append(current)
        if current == goal_idx:
            return True
        for nb in neighbors(current, width, height):
            if nb == prev:
                continue
            if final_values[nb] != color:
                continue
            if dfs(nb, current):
                return True
        path.pop()
        return False

    if not dfs(start_idx, None):
        raise ValueError(f"Unable to trace path for color {color}")
    if set(path) != set(color_cells):
        raise ValueError(f"Path for color {color} does not cover all cells")
    return path


def solution_values_from_string(
    puzzle_tokens: list[str],
    solved_tokens: list[str],
    width: int,
    height: int,
    color_count: int,
) -> list[int]:
    """Convert solved board tokens to color values respecting original endpoints.

    The solved board may use different color numbering than the puzzle, so this
    function remaps colors to match the original puzzle's endpoint colors.

    Args:
        puzzle_tokens: Original puzzle tokens
        solved_tokens: Solution tokens
        width: Grid width
        height: Grid height
        color_count: Number of colors

    Returns:
        List of integer color values (flat, row-major order)

    Raises:
        ValueError: If solution doesn't respect endpoints or has invalid mapping
    """
    if len(solved_tokens) != width * height:
        raise ValueError("Solved board length mismatch")

    # Build bidirectional mapping between solved and puzzle colors
    mapping: dict[int, int] = {}  # solved_color -> puzzle_color
    reverse: dict[int, int] = {}  # puzzle_color -> solved_color

    for idx, tok in enumerate(puzzle_tokens):
        if tok.lower() == "x":
            continue
        s_val = int(solved_tokens[idx])
        p_val = int(tok)

        # Verify consistent mapping
        if s_val in mapping and mapping[s_val] != p_val:
            raise ValueError("Solved board does not respect puzzle endpoints")
        if p_val in reverse and reverse[p_val] != s_val:
            raise ValueError("Solved board does not respect puzzle endpoints")

        mapping[s_val] = p_val
        reverse[p_val] = s_val

    # Remap all solved values to puzzle colors
    remapped: list[int] = []
    for tok in solved_tokens:
        val = int(tok)
        remapped.append(mapping.get(val, val))
    return remapped


__all__ = [
    "extract_endpoints",
    "build_color_path",
    "solution_values_from_string",
]
