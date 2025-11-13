from __future__ import annotations

from dataclasses import dataclass

from rl.env.config import RewardPreset
from rl.env.rewards.base import RewardEngine
from rl.env.rewards.composite import CompositeReward
from rl.env.rewards.completion import CompletionReward
from rl.env.rewards.constraints import ConstraintPenalty
from rl.env.rewards.potential import PotentialReward


@dataclass
class RewardSystem:
    engine: RewardEngine
    invalid_penalty: float
    unsolved_penalty: float
    undo_penalty: float
    solve_efficiency_bonus: float = 0.0  # Bonus per step remaining when solved


def build_reward_system(preset: RewardPreset) -> RewardSystem:
    params = preset.params
    engines: list[RewardEngine] = []
    invalid_penalty = params.get("invalid_penalty", 0.0)
    unsolved_penalty = params.get("unsolved_penalty", 0.0)
    undo_penalty = params.get("undo_penalty", 0.0)
    solve_efficiency_bonus = params.get("solve_efficiency_bonus", 0.0)

    for component in preset.components:
        if component == "potential":
            engines.append(
                PotentialReward(
                    move_penalty=params.get("move_penalty", 0.0),
                    distance_scale=params.get("distance_scale", 0.0),
                )
            )
        elif component == "completion":
            engines.append(
                CompletionReward(
                    complete_bonus=params.get("complete_bonus", 0.0),
                    solve_bonus=params.get("solve_bonus", 0.0),
                    revert_penalty=params.get("complete_revert_penalty"),
                    sustain_bonus=params.get("complete_sustain_bonus", 0.0),
                )
            )
        elif component == "constraints":
            engines.append(
                ConstraintPenalty(
                    dead_pocket_penalty=params.get("dead_pocket_penalty", 0.0),
                    disconnect_penalty=params.get("disconnect_penalty", 0.0),
                    degree_penalty=params.get("degree_penalty", 0.0),
                )
            )
        else:
            raise ValueError(f"Unknown reward component '{component}' in preset '{preset.name}'")

    return RewardSystem(
        engine=CompositeReward(engines=engines),
        invalid_penalty=invalid_penalty,
        unsolved_penalty=unsolved_penalty,
        undo_penalty=undo_penalty,
        solve_efficiency_bonus=solve_efficiency_bonus,
    )


__all__ = ["RewardSystem", "build_reward_system"]
