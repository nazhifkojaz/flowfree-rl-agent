"""Longest path first completion strategy."""

from __future__ import annotations

from rl.env.trace_generation.strategies.base import CompletionContext, CompletionStrategy


class LongestPathStrategy(CompletionStrategy):
    """Complete colors in descending order of path length.

    Colors with longer paths are completed first, potentially creating more
    diverse training examples.
    """

    def build_color_order(self, ctx: CompletionContext) -> list[int]:
        """Return colors sorted by path length (longest first)."""
        colors = sorted(ctx.paths.keys())
        return sorted(colors, key=lambda c: len(ctx.paths[c]), reverse=True)

    def build_edge_schedule(
        self, order: list[int], ctx: CompletionContext
    ) -> list[tuple[int, int]]:
        """Schedule all edges of each color sequentially."""
        schedule: list[tuple[int, int]] = []
        for color in order:
            path = ctx.paths.get(color, [])
            if len(path) < 2:
                continue
            for edge_idx in range(len(path) - 1):
                schedule.append((color, edge_idx))
        return schedule

    def supports_variants(self) -> bool:
        """Longest path strategy uses fixed order."""
        return False


__all__ = ["LongestPathStrategy"]
