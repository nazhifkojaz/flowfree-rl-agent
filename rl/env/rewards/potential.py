from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from rl.env.rewards.base import RewardContext, RewardEngine, RewardResult


@dataclass
class PotentialReward(RewardEngine):
    """
    Potential-based shaping using Manhattan distance to the target plus a move penalty.
    """

    move_penalty: float
    distance_scale: float

    def compute(self, ctx: RewardContext) -> RewardResult:
        if ctx.diff is None:
            return RewardResult(value=0.0, breakdown={})

        breakdown: Dict[str, float] = {}
        reward = 0.0

        if self.move_penalty and not ctx.diff.undone:
            reward += self.move_penalty
            breakdown["move_penalty"] = self.move_penalty

        color = ctx.diff.color
        prev_dist = ctx.previous.distance_to_target[color]
        next_dist = ctx.next_state.distance_to_target[color]
        delta = prev_dist - next_dist
        if self.distance_scale and delta > 0:
            shaped = self.distance_scale * delta
            reward += shaped
            breakdown["distance"] = shaped

        return RewardResult(value=reward, breakdown=breakdown)
