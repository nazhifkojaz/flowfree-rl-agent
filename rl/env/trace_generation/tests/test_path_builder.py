"""Unit tests for path building functions."""

import pytest

from rl.env.trace_generation.path_builder import (
    build_color_path,
    extract_endpoints,
    solution_values_from_string,
)


def test_extract_endpoints():
    """Test endpoint extraction from puzzle."""
    puzzle = ["1", "x", "1", "2", "x", "2"]
    endpoints = extract_endpoints(puzzle, width=3, height=2, color_count=2)
    assert endpoints == {1: (0, 2), 2: (3, 5)}


def test_extract_endpoints_incomplete():
    """Test endpoint extraction with incomplete color."""
    puzzle = ["1", "x", "1", "2", "x", "x"]  # Color 2 has only one endpoint
    endpoints = extract_endpoints(puzzle, width=3, height=2, color_count=2)
    assert 1 in endpoints
    assert 2 not in endpoints  # Incomplete color excluded


def test_build_color_path_simple():
    """Test simple path reconstruction."""
    final_values = [1, 1, 1, 2, 2, 2]
    path = build_color_path(1, endpoints=(0, 2), final_values=final_values, width=3, height=2)
    assert path == [0, 1, 2]


def test_build_color_path_with_turn():
    """Test path with a turn."""
    # Grid:
    # 1 1 2
    # x 1 2
    final_values = [1, 1, 2, 0, 1, 2]
    path = build_color_path(1, endpoints=(0, 4), final_values=final_values, width=3, height=2)
    assert set(path) == {0, 1, 4}
    assert path[0] == 0 and path[-1] == 4


def test_build_color_path_invalid():
    """Test path reconstruction with disconnected cells."""
    final_values = [1, 0, 1, 2, 2, 2]  # Color 1 disconnected
    with pytest.raises(ValueError, match="Unable to trace path"):
        build_color_path(1, endpoints=(0, 2), final_values=final_values, width=3, height=2)


def test_solution_values_from_string():
    """Test solution value remapping."""
    puzzle_tokens = ["1", "x", "1", "2", "x", "2"]
    solved_tokens = ["2", "2", "2", "1", "1", "1"]  # Colors swapped
    remapped = solution_values_from_string(
        puzzle_tokens, solved_tokens, width=3, height=2, color_count=2
    )
    # Should remap back to puzzle colors
    assert remapped == [1, 1, 1, 2, 2, 2]


def test_solution_values_invalid_length():
    """Test solution validation with length mismatch."""
    puzzle_tokens = ["1", "x", "1"]
    solved_tokens = ["1", "1"]  # Too short
    with pytest.raises(ValueError, match="length mismatch"):
        solution_values_from_string(
            puzzle_tokens, solved_tokens, width=3, height=1, color_count=1
        )


def test_solution_values_endpoint_violation():
    """Test solution validation with endpoint mismatch."""
    puzzle_tokens = ["1", "x", "1"]
    solved_tokens = ["1", "2", "2"]  # Wrong color at endpoint
    with pytest.raises(ValueError, match="does not respect puzzle endpoints"):
        solution_values_from_string(
            puzzle_tokens, solved_tokens, width=3, height=1, color_count=2
        )
