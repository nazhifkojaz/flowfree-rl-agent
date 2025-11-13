"""Completion strategies for trace generation.

This package provides different strategies for determining the order in which
colors are completed when replaying solved puzzles.
"""

from rl.env.trace_generation.strategies.base import CompletionContext, CompletionStrategy
from rl.env.trace_generation.strategies.blocked import BlockedStrategy
from rl.env.trace_generation.strategies.longest import LongestPathStrategy
from rl.env.trace_generation.strategies.normal import NormalStrategy
from rl.env.trace_generation.strategies.one_at_time import OneAtTimeStrategy
from rl.env.trace_generation.strategies.one_distance import OneDistanceStrategy

# Strategy registry
STRATEGIES: dict[str, type[CompletionStrategy]] = {
    "normal": NormalStrategy,
    "longest": LongestPathStrategy,
    "onedistance": OneDistanceStrategy,
    "blocked": BlockedStrategy,
    "oneattime": OneAtTimeStrategy,
}


def get_strategy(mode: str) -> CompletionStrategy:
    """Get a completion strategy instance by name.

    Args:
        mode: Strategy name (normal, longest, onedistance, blocked, oneattime)

    Returns:
        CompletionStrategy instance

    Raises:
        KeyError: If strategy name is not recognized
    """
    if mode not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise KeyError(f"Unknown strategy '{mode}'. Available strategies: {available}")
    return STRATEGIES[mode]()


__all__ = [
    "CompletionStrategy",
    "CompletionContext",
    "NormalStrategy",
    "LongestPathStrategy",
    "OneDistanceStrategy",
    "BlockedStrategy",
    "OneAtTimeStrategy",
    "STRATEGIES",
    "get_strategy",
]
