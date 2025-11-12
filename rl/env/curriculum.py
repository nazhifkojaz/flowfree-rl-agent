from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque


@dataclass
class CurriculumEvent:
    kind: str
    payload: dict


@dataclass
class SuccessRateCurriculum:
    """
    Tracks recent episode outcomes and triggers callbacks when success rate crosses thresholds.

    Example usage:
        curriculum = SuccessRateCurriculum(
            window=20,
            promote_threshold=0.8,
            demote_threshold=0.2,
            on_promote=lambda: increase_board_size(),
        )
    """

    window: int
    promote_threshold: float
    demote_threshold: float
    on_promote: Callable[[], None]
    on_demote: Callable[[], None] | None = None
    history: Deque[int] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def update(self, terminated: bool) -> CurriculumEvent | None:
        self.history.append(1 if terminated else 0)
        if len(self.history) < self.window:
            return None

        rate = sum(self.history) / len(self.history)
        if rate >= self.promote_threshold:
            self.on_promote()
            return CurriculumEvent(kind="promote", payload={"success_rate": rate})
        if self.on_demote and rate <= self.demote_threshold:
            self.on_demote()
            return CurriculumEvent(kind="demote", payload={"success_rate": rate})
        return None


__all__ = ["SuccessRateCurriculum", "CurriculumEvent"]

