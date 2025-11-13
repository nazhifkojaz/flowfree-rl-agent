"""Blocked completion strategy: prioritize colors with fewer remaining edges."""

from __future__ import annotations

from rl.env.trace_generation.strategies.base import CompletionContext, CompletionStrategy


class BlockedStrategy(CompletionStrategy):
    """Complete colors by repeatedly choosing the one with fewest edges left.

    This strategy dynamically selects which color to extend next based on how many
    edges each color still has remaining, prioritizing colors that are closer to
    completion. This can create more complex interleaving patterns.
    """

    def build_color_order(self, ctx: CompletionContext) -> list[int]:
        """Return colors in ascending order (used as base priority)."""
        return sorted(ctx.paths.keys())

    def build_edge_schedule(
        self, order: list[int], ctx: CompletionContext
    ) -> list[tuple[int, int]]:
        """Greedily schedule the color with fewest remaining edges at each step."""
        # Track remaining edges for each color
        available = {
            color: max(0, len(ctx.paths.get(color, [])) - 1) for color in order
        }

        schedule: list[tuple[int, int]] = []
        while any(available.values()):
            # Find color with fewest remaining edges (ties broken by order index)
            constraints = []
            for idx, color in enumerate(order):
                remaining = available[color]
                if remaining <= 0:
                    continue
                constraints.append((remaining, idx, color))

            if not constraints:
                break

            # Choose color with minimum remaining edges
            _, _, chosen = min(constraints)

            # Compute which edge to place next for this color
            path = ctx.paths[chosen]
            edges_placed = len(path) - available[chosen] - 1
            schedule.append((chosen, edges_placed))

            # Update remaining count
            available[chosen] -= 1

        return schedule

    def supports_variants(self) -> bool:
        """Blocked strategy uses deterministic greedy order."""
        return False


__all__ = ["BlockedStrategy"]
