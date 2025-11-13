"""Unit tests for completion strategies."""

import pytest

from rl.env.trace_generation.strategies import (
    BlockedStrategy,
    CompletionContext,
    LongestPathStrategy,
    NormalStrategy,
    OneAtTimeStrategy,
    OneDistanceStrategy,
    get_strategy,
)


@pytest.fixture
def sample_context():
    """Sample completion context for testing."""
    return CompletionContext(
        paths={
            1: [0, 1, 2],  # 3 cells = 2 edges
            2: [3, 4],  # 2 cells = 1 edge
            3: [5, 6, 7, 8],  # 4 cells = 3 edges
        },
        endpoints={
            1: (0, 2),
            2: (3, 4),
            3: (5, 8),
        },
        width=3,
        height=3,
        color_count=3,
    )


def test_get_strategy():
    """Test strategy registry."""
    assert isinstance(get_strategy("normal"), NormalStrategy)
    assert isinstance(get_strategy("longest"), LongestPathStrategy)
    assert isinstance(get_strategy("onedistance"), OneDistanceStrategy)
    assert isinstance(get_strategy("blocked"), BlockedStrategy)
    assert isinstance(get_strategy("oneattime"), OneAtTimeStrategy)

    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("invalid")


def test_normal_strategy(sample_context):
    """Test normal completion strategy."""
    strategy = NormalStrategy()
    order = strategy.build_color_order(sample_context)
    assert order == [1, 2, 3]  # Ascending order

    schedule = strategy.build_edge_schedule(order, sample_context)
    assert len(schedule) == 6  # 2 + 1 + 3 edges (N cells = N-1 edges)
    assert schedule[0] == (1, 0)  # First edge of color 1
    assert schedule[1] == (1, 1)  # Second edge of color 1
    assert schedule[2] == (2, 0)  # First edge of color 2
    assert strategy.supports_variants()


def test_longest_path_strategy(sample_context):
    """Test longest path first strategy."""
    strategy = LongestPathStrategy()
    order = strategy.build_color_order(sample_context)
    assert order == [3, 1, 2]  # Descending by path length: 4, 3, 2

    schedule = strategy.build_edge_schedule(order, sample_context)
    assert len(schedule) == 6  # 3 + 2 + 1 edges
    assert schedule[0] == (3, 0)  # Color 3 first
    assert not strategy.supports_variants()


def test_one_distance_strategy(sample_context):
    """Test one distance strategy."""
    strategy = OneDistanceStrategy()
    order = strategy.build_color_order(sample_context)
    # Distances: color 1 = 2, color 2 = 1, color 3 = 1
    # Sort descending by distance, ties broken by ascending color number
    assert order == [1, 2, 3]  # Color 1 has largest distance (2)

    schedule = strategy.build_edge_schedule(order, sample_context)
    assert len(schedule) == 6  # 2 + 1 + 3 edges
    assert not strategy.supports_variants()


def test_one_at_time_strategy(sample_context):
    """Test one at a time interleaving strategy."""
    strategy = OneAtTimeStrategy()
    order = strategy.build_color_order(sample_context)
    assert order == [1, 2, 3]

    schedule = strategy.build_edge_schedule(order, sample_context)
    assert len(schedule) == 6  # 2 + 1 + 3 edges total
    # Should interleave: edge 0 of all, then edge 1 of all, etc.
    assert schedule[0] == (1, 0)  # edge 0 of color 1
    assert schedule[1] == (2, 0)  # edge 0 of color 2
    assert schedule[2] == (3, 0)  # edge 0 of color 3
    assert schedule[3] == (1, 1)  # edge 1 of color 1
    # Color 2 has no edge 1, skip
    assert schedule[4] == (3, 1)  # edge 1 of color 3
    assert not strategy.supports_variants()


def test_blocked_strategy(sample_context):
    """Test blocked (greedy minimum) strategy."""
    strategy = BlockedStrategy()
    order = strategy.build_color_order(sample_context)
    assert order == [1, 2, 3]

    schedule = strategy.build_edge_schedule(order, sample_context)
    assert len(schedule) == 6  # 2 + 1 + 3 edges total
    # Greedy picks color with fewest remaining edges each time
    # Initial: 1 has 2, 2 has 1, 3 has 3 → pick 2
    assert schedule[0] == (2, 0)
    # After: 1 has 2, 3 has 3 → pick 1
    assert schedule[1] == (1, 0)
    assert not strategy.supports_variants()
