from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardSettings:
    move_penalty: float = -0.05
    distance_bonus: float = 0.35
    invalid_penalty: float = -1.0
    complete_color_bonus: float = 1.0
    solve_bonus: float = 20.0
    unsolved_penalty: float = -2.0
    dead_pocket_penalty: float = 0.0
    disconnect_penalty: float = -0.08
    degree_penalty: float = -0.08
    segment_connection_bonus: float = 0.0
    path_extension_bonus: float = 0.0
    move_reduction_bonus: float = 0.0
    dead_end_penalty: float = 0.0
    distance_penalty: float = 0.0
    cell_fill_bonus: float = 0.0
    color_switch_penalty: float = 0.0
    streak_bonus_per_move: float = 0.0
    streak_length: int = 2
    loop_penalty: float = 0.0
    loop_window: int = 0
    progress_bonus: float = 0.0


def get_simple_reward_settings() -> RewardSettings:
    return RewardSettings(
        solve_bonus=25.0,
        complete_color_bonus=2.0,
        distance_bonus=0.4,
        move_penalty=-0.05,
        invalid_penalty=-1.0,
        disconnect_penalty=-0.5,
        degree_penalty=-0.3,
        dead_pocket_penalty=-0.5,
        unsolved_penalty=-5.0,
        segment_connection_bonus=0.0,
        path_extension_bonus=0.0,
        move_reduction_bonus=0.0,
        dead_end_penalty=0.0,
        distance_penalty=0.0,
        cell_fill_bonus=0.0,
        color_switch_penalty=0.0,
        streak_bonus_per_move=0.0,
    )


__all__ = ["RewardSettings", "get_simple_reward_settings"]
