"""Normal completion strategy: complete each color sequentially."""

from __future__ import annotations

from rl.env.trace_generation.strategies.base import CompletionContext, CompletionStrategy


class NormalStrategy(CompletionStrategy):
    """Complete colors in ascending order, one at a time.

    This is the default strategy where each color's path is completed fully
    before moving to the next color.
    """

    def build_color_order(self, ctx: CompletionContext) -> list[int]:
        """Return colors in ascending order."""
        return sorted(ctx.paths.keys())

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
        """Normal strategy supports color order shuffling."""
        return True


__all__ = ["NormalStrategy"]
