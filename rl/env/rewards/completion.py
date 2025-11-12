from __future__ import annotations

from dataclasses import dataclass

from rl.env.rewards.base import RewardContext, RewardEngine, RewardResult


@dataclass
class CompletionReward(RewardEngine):
    complete_bonus: float
    solve_bonus: float
    revert_penalty: float | None = None
    sustain_bonus: float = 0.0

    def compute(self, ctx: RewardContext) -> RewardResult:
        if ctx.diff is None:
            return RewardResult(value=0.0, breakdown={})

        reward = 0.0
        breakdown: dict[str, float] = {}

        # Handle undoing a previously completed colour/board.
        if ctx.diff.undone:
            if ctx.diff.reverted_completion:
                penalty = self.revert_penalty if self.revert_penalty is not None else self.complete_bonus
                if penalty:
                    reward -= penalty
                    breakdown["complete_undo"] = -penalty
            if ctx.previous.all_completed() and not ctx.next_state.all_completed() and self.solve_bonus:
                reward -= self.solve_bonus
                breakdown["solve_undo"] = -self.solve_bonus
            return RewardResult(value=reward, breakdown=breakdown)

        if ctx.diff.completed_now and self.complete_bonus:
            total_colors = ctx.next_state.shape.color_count
            completed_count = sum(1 for flag in ctx.next_state.completed if flag)
            remaining = max(0, total_colors - completed_count)
            ratio = (remaining / total_colors) if total_colors > 0 else 0.0
            bonus = self.complete_bonus * (1.0 + ratio)
            reward += bonus
            breakdown["complete"] = bonus

        if ctx.next_state.all_completed() and self.solve_bonus:
            reward += self.solve_bonus
            breakdown["solve"] = self.solve_bonus

        if self.sustain_bonus:
            completed_count = sum(1 for flag in ctx.next_state.completed if flag)
            if completed_count:
                sustain = self.sustain_bonus * completed_count
                reward += sustain
                breakdown["complete_sustain"] = sustain

        return RewardResult(value=reward, breakdown=breakdown)
