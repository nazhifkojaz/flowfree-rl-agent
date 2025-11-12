from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rl.env.rewards.base import RewardContext, RewardEngine, RewardResult


@dataclass
class CompositeReward(RewardEngine):
    engines: Iterable[RewardEngine]

    def compute(self, ctx: RewardContext) -> RewardResult:
        value = 0.0
        breakdown: dict[str, float] = {}

        for engine in self.engines:
            result = engine.compute(ctx)
            value += result.value
            for key, delta in result.breakdown.items():
                breakdown[key] = breakdown.get(key, 0.0) + delta

        return RewardResult(value=value, breakdown=breakdown)

