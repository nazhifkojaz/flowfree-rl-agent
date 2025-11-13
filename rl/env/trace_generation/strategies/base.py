"""Base class for completion strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionContext:
    """Shared data needed by all completion strategies.

    Attributes:
        paths: Dict mapping color (1-indexed) to list of cell indices forming its path
        endpoints: Dict mapping color to (start_idx, end_idx) tuple
        width: Grid width
        height: Grid height
        color_count: Total number of colors in the puzzle
    """

    paths: dict[int, list[int]]
    endpoints: dict[int, tuple[int, int]]
    width: int
    height: int
    color_count: int


class CompletionStrategy(ABC):
    """Abstract base class for color completion order strategies.

    Subclasses implement different strategies for determining the order in which
    colors are completed and how edges are scheduled for replay.
    """

    @abstractmethod
    def build_color_order(self, ctx: CompletionContext) -> list[int]:
        """Determine the order in which colors should be completed.

        Args:
            ctx: Context with paths, endpoints, and grid dimensions

        Returns:
            List of color indices (1-indexed) in completion order
        """
        pass

    @abstractmethod
    def build_edge_schedule(
        self, order: list[int], ctx: CompletionContext
    ) -> list[tuple[int, int]]:
        """Build a schedule of (color, edge_idx) tuples for replay.

        Args:
            order: Color completion order (from build_color_order)
            ctx: Context with paths, endpoints, and grid dimensions

        Returns:
            List of (color, edge_idx) tuples defining the move sequence
        """
        pass

    def supports_variants(self) -> bool:
        """Whether this strategy supports multiple variants via shuffling.

        Returns:
            True if color order can be shuffled to create variants
        """
        return False


__all__ = ["CompletionStrategy", "CompletionContext"]
