from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from dualstream_agent.schemas import ReasoningAction, ResponseAction, RuntimeOutput


@dataclass(slots=True)
class HarnessMetrics:
    total_events: int = 0
    responses: int = 0
    waits: int = 0
    interrupts: int = 0
    s2_invocations: int = 0
    response_latencies: list[float] = field(default_factory=list)
    expected_response_hits: int = 0
    expected_response_misses: int = 0
    false_responses: int = 0

    def observe(
        self,
        output: RuntimeOutput,
        *,
        event_started_at: float,
        expect_response: bool | None = None,
    ) -> None:
        self.total_events += 1
        responded = output.decision.response_action in {
            ResponseAction.RESPOND,
            ResponseAction.INTERRUPT,
        }
        if responded:
            self.responses += 1
            self.response_latencies.append(max(0.0, output.timestamp - event_started_at))
        else:
            self.waits += 1
        if output.decision.response_action == ResponseAction.INTERRUPT:
            self.interrupts += 1
        if output.decision.reasoning_action != ReasoningAction.S1_ONLY:
            self.s2_invocations += 1
        if expect_response is True:
            if responded:
                self.expected_response_hits += 1
            else:
                self.expected_response_misses += 1
        elif expect_response is False and responded:
            self.false_responses += 1

    def summary(self) -> dict[str, float | int]:
        expected_total = self.expected_response_hits + self.expected_response_misses
        return {
            "total_events": self.total_events,
            "responses": self.responses,
            "waits": self.waits,
            "interrupts": self.interrupts,
            "s2_invocations": self.s2_invocations,
            "mean_response_latency": mean(self.response_latencies)
            if self.response_latencies
            else 0.0,
            "response_recall": self.expected_response_hits / expected_total
            if expected_total
            else 0.0,
            "false_response_rate": self.false_responses / self.total_events
            if self.total_events
            else 0.0,
        }
