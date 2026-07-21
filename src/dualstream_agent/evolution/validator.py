from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from dualstream_agent.skills.manager import SkillManager

EvaluationFn = Callable[[str | None], Awaitable[dict[str, float]]]


@dataclass(slots=True)
class ValidationResult:
    candidate: str
    baseline_score: float
    candidate_score: float
    latency_delta: float
    regression_rate: float
    utility: float
    promoted: bool


class ReplayValidator:
    """Compare a candidate skill against a no-candidate baseline."""

    def __init__(
        self,
        skills: SkillManager,
        *,
        latency_weight: float = 0.05,
        regression_weight: float = 0.5,
        minimum_gain: float = 0.01,
    ):
        self.skills = skills
        self.latency_weight = latency_weight
        self.regression_weight = regression_weight
        self.minimum_gain = minimum_gain

    async def validate(self, candidate: str, evaluate: EvaluationFn) -> ValidationResult:
        baseline = await evaluate(None)
        treatment = await evaluate(candidate)
        baseline_score = float(baseline.get("score", 0.0))
        candidate_score = float(treatment.get("score", 0.0))
        latency_delta = float(treatment.get("latency_s", 0.0)) - float(
            baseline.get("latency_s", 0.0)
        )
        regression_rate = float(treatment.get("regression_rate", 0.0))
        utility = (
            candidate_score
            - baseline_score
            - self.latency_weight * max(0.0, latency_delta)
            - self.regression_weight * regression_rate
        )
        promoted = utility >= self.minimum_gain
        result = ValidationResult(
            candidate=candidate,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            latency_delta=latency_delta,
            regression_rate=regression_rate,
            utility=utility,
            promoted=promoted,
        )
        if promoted:
            self.skills.promote(candidate, asdict(result))
        return result
