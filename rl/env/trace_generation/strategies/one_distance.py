"""One distance completion strategy: complete by Manhattan distance."""

from __future__ import annotations

from rl.env.trace_generation.strategies.base import CompletionContext, CompletionStrategy


class OneDistanceStrategy(CompletionStrategy):
    """Complete colors in descending order of endpoint Manhattan distance.

    Colors whose endpoints are farther apart (in Manhattan distance) are
    completed first.
    """

    def build_color_order(self, ctx: CompletionContext) -> list[int]:
        """Return colors sorted by endpoint Manhattan distance (largest first)."""

        def manhattan(color: int) -> int:
            """Compute Manhattan distance between color's endpoints."""
            endpoints = ctx.endpoints.get(color)
            if endpoints is None:
                return 0
            start, end = endpoints
            sr, sc = divmod(start, ctx.width)
            er, ec = divmod(end, ctx.width)
            return abs(sr - er) + abs(sc - ec)

        colors = sorted(ctx.paths.keys())
        return sorted(colors, key=manhattan, reverse=True)

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
        """One distance strategy uses fixed order."""
        return False


__all__ = ["OneDistanceStrategy"]
