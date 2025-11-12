"""Utility functions for the FlowFree RL environment."""

from functools import lru_cache


def string_to_tokens(
    puzzle_str: str,
    width: int,
    height: int,
    color_count: int
) -> list[str] | None:
    """Parse a puzzle string into tokens.

    Converts compact puzzle strings like "1x2x3..." into a list of tokens.
    Each token is either 'x' (empty cell) or a color number string.

    Args:
        puzzle_str: The puzzle string to parse
        width: Board width
        height: Board height
        color_count: Maximum number of colors

    Returns:
        List of tokens if valid, None if parsing fails
    """
    s = puzzle_str.strip()
    n = len(s)
    target = width * height # number of tokens/cells expected
    max_digits = len(str(color_count))

    @lru_cache(maxsize=None)
    def parse(i: int, tcount: int):
        """
        Return a list of tokens from s[i:] that yields exactly (target - tcount)
        remaining tokens, or None if impossible. Tokens are 'x' or valid color strings.
        """
        # invalid structure/mismatch shape??
        if tcount > target:
            return None

        # exit when target is reached
        if i == n:
            return [] if tcount == target else None

        ch = s[i]

        # Empty cells
        if ch in ("x", "X"):
            rest = parse(i + 1, tcount + 1)
            return (["x"] + rest) if rest is not None else None

        # not a digit -> invalid
        if not ch.isdigit():
            return None

        # colors are labelled 1~color_count, so no zeros
        if ch == "0":
            return None

        # capture the longest possible combination till we hit 'x' or end of puzzle string
        j = i
        while j < n and s[j].isdigit():
            j += 1
        run = s[i:j]

        # clamp to max_digits
        len_max = min(max_digits, len(run))

        # try longest numeric combination first,
        # but backtrack if it doesn't lead to a correct shape/target
        for L in range(len_max, 0, -1):
            tok = run[:L]
            # Leading zeros already excluded by ch != '0'
            val = int(tok)
            if 1 <= val <= color_count:
                rest = parse(i + L, tcount + 1)
                if rest is not None:
                    return [tok] + rest

        return None

    return parse(0, 0)


__all__ = ["string_to_tokens"]
