"""Trace generation package for creating RL training data from solved puzzles.

This package provides utilities to generate expert demonstration trajectories
by replaying completed Flow Free puzzles with various completion strategies.
"""

from rl.env.trace_generation.config import TRACE_REWARD, TraceGenConfig
from rl.env.trace_generation.grid_utils import encode_move, hash_puzzle, neighbors
from rl.env.trace_generation.io import read_rows
from rl.env.trace_generation.path_builder import (
    build_color_path,
    extract_endpoints,
    solution_values_from_string,
)

__all__ = [
    "TraceGenConfig",
    "TRACE_REWARD",
    "neighbors",
    "encode_move",
    "hash_puzzle",
    "extract_endpoints",
    "build_color_path",
    "solution_values_from_string",
    "read_rows",
]
