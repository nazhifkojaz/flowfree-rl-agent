"""Unit tests for grid utility functions."""

import pytest

from rl.env.trace_generation.grid_utils import encode_move, hash_puzzle, neighbors


def test_neighbors_center():
    """Test neighbor calculation for center cell."""
    # 3x3 grid, center cell (index 4)
    nbrs = neighbors(4, width=3, height=3)
    assert set(nbrs) == {1, 3, 5, 7}  # up, left, right, down


def test_neighbors_corner():
    """Test neighbor calculation for corner cell."""
    # 3x3 grid, top-left corner (index 0)
    nbrs = neighbors(0, width=3, height=3)
    assert set(nbrs) == {1, 3}  # right, down only


def test_neighbors_edge():
    """Test neighbor calculation for edge cell."""
    # 3x3 grid, top edge (index 1)
    nbrs = neighbors(1, width=3, height=3)
    assert set(nbrs) == {0, 2, 4}  # left, right, down


def test_encode_move_up():
    """Test encoding upward move."""
    action = encode_move(from_idx=4, to_idx=1, color=1, width=3)
    assert action == 0  # UP direction


def test_encode_move_right():
    """Test encoding rightward move."""
    action = encode_move(from_idx=4, to_idx=5, color=1, width=3)
    assert action == 1  # RIGHT direction


def test_encode_move_down():
    """Test encoding downward move."""
    action = encode_move(from_idx=4, to_idx=7, color=1, width=3)
    assert action == 2  # DOWN direction


def test_encode_move_left():
    """Test encoding leftward move."""
    action = encode_move(from_idx=4, to_idx=3, color=1, width=3)
    assert action == 3  # LEFT direction


def test_encode_move_multiple_colors():
    """Test encoding with different colors."""
    action1 = encode_move(from_idx=0, to_idx=1, color=1, width=3)
    action2 = encode_move(from_idx=0, to_idx=1, color=2, width=3)
    action3 = encode_move(from_idx=0, to_idx=1, color=3, width=3)
    # Verify actions are different for different colors
    assert action1 != action2 != action3
    # Each color adds 5 to base action (ACTIONS_PER_COLOR = 5)
    from rl.env.constants import ACTIONS_PER_COLOR
    assert action2 == action1 + ACTIONS_PER_COLOR
    assert action3 == action1 + (2 * ACTIONS_PER_COLOR)


def test_encode_move_invalid():
    """Test encoding non-adjacent move."""
    with pytest.raises(ValueError, match="not axis-aligned"):
        encode_move(from_idx=0, to_idx=4, color=1, width=3)  # Diagonal


def test_hash_puzzle():
    """Test puzzle hashing."""
    puzzle = "11xx2x32"
    hash1 = hash_puzzle(puzzle)
    assert len(hash1) == 12
    assert hash1 == hash_puzzle(puzzle)  # Deterministic

    # Different puzzle should have different hash
    puzzle2 = "22xx1x31"
    hash2 = hash_puzzle(puzzle2)
    assert hash1 != hash2
