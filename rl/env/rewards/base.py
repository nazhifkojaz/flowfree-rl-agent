from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

from rl.env.state import BoardState, StateDiff


@dataclass(frozen=True)
class RewardContext:
    previous: BoardState
    next_state: BoardState
    diff: StateDiff | None
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class RewardResult:
    value: float
    breakdown: Dict[str, float]


class RewardEngine(Protocol):
    def compute(self, ctx: RewardContext) -> RewardResult:
        ... # override in implementations

