"""One at a time completion strategy: interleave edges across colors."""

from __future__ import annotations

from rl.env.trace_generation.strategies.base import CompletionContext, CompletionStrategy


class OneAtTimeStrategy(CompletionStrategy):
    """Interleave completion by placing one edge per color at a time.

    Rather than completing each color fully, this strategy places the first edge
    of all colors, then the second edge of all colors, etc., creating a more
    interleaved completion pattern.
    """

    def build_color_order(self, ctx: CompletionContext) -> list[int]:
        """Return colors in ascending order."""
        return sorted(ctx.paths.keys())

    def build_edge_schedule(
        self, order: list[int], ctx: CompletionContext
    ) -> list[tuple[int, int]]:
        """Interleave edges: place edge_idx=0 for all colors, then edge_idx=1, etc."""
        # Find maximum path length
        max_edges = 0
        for color in order:
            path = ctx.paths.get(color, [])
            max_edges = max(max_edges, max(0, len(path) - 1))

        # Schedule edges in rounds
        schedule: list[tuple[int, int]] = []
        for edge_idx in range(max_edges):
            for color in order:
                path = ctx.paths.get(color, [])
                if edge_idx < len(path) - 1:
                    schedule.append((color, edge_idx))

        return schedule

    def supports_variants(self) -> bool:
        """One at a time strategy uses fixed order."""
        return False


__all__ = ["OneAtTimeStrategy"]
