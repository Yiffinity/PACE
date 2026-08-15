from __future__ import annotations

from dataclasses import dataclass


class ExperienceContextOverflowError(RuntimeError):
    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget
        super().__init__(
            "experience context overflow: "
            f"required={budget.required_tokens}, limit={budget.context_limit}, "
            f"experiences={budget.experience_tokens}, sample={budget.sample_id}"
        )


@dataclass(frozen=True, slots=True)
class ContextBudget:
    sample_id: str
    context_limit: int
    system_tokens: int
    image_tokens: int
    text_tokens: int
    experience_tokens: int
    prefix_tokens: int
    response_reserve_tokens: int
    safety_margin_tokens: int

    @property
    def required_tokens(self) -> int:
        return (
            self.system_tokens
            + self.image_tokens
            + self.text_tokens
            + self.experience_tokens
            + self.prefix_tokens
            + self.response_reserve_tokens
            + self.safety_margin_tokens
        )

    @property
    def utilization(self) -> float:
        return self.required_tokens / self.context_limit

    def warning_thresholds_crossed(self, thresholds: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(threshold for threshold in thresholds if self.utilization >= threshold)

    def ensure_fits(self) -> None:
        if self.required_tokens > self.context_limit:
            raise ExperienceContextOverflowError(self)
