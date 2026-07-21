from __future__ import annotations

from dataclasses import dataclass

from dualstream_agent.config import ControllerConfig
from dualstream_agent.schemas import (
    ControllerDecision,
    ReasoningAction,
    ResponseAction,
    S1Signal,
)


@dataclass(slots=True)
class ControllerState:
    last_response_at: float | None = None
    speaking: bool = False


class DualController:
    """Rule-constrained dual-system scheduler with cooldown and hysteresis."""

    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.state = ControllerState()

    def decide(self, signal: S1Signal, now: float) -> ControllerDecision:
        combined = 0.45 * signal.relevance + 0.35 * signal.novelty + 0.20 * signal.confidence
        cooling_down = (
            self.state.last_response_at is not None
            and now - self.state.last_response_at < self.config.cooldown_s
        )

        if signal.urgency >= self.config.interrupt_threshold:
            self.state.last_response_at = now
            return ControllerDecision(
                response_action=ResponseAction.INTERRUPT,
                reasoning_action=ReasoningAction.S1_ONLY,
                reason="urgency_above_interrupt_threshold",
                score=signal.urgency,
            )

        if signal.need_tool:
            return ControllerDecision(
                response_action=ResponseAction.WAIT,
                reasoning_action=ReasoningAction.INVOKE_TOOL,
                reason="s1_requested_tool",
                score=combined,
            )

        if signal.need_reasoning or signal.need_memory or signal.confidence < self.config.confidence_threshold:
            action = (
                ReasoningAction.RETRIEVE_MEMORY
                if signal.need_memory and not signal.need_tool
                else ReasoningAction.INVOKE_S2
            )
            return ControllerDecision(
                response_action=ResponseAction.WAIT,
                reasoning_action=action,
                reason="slow_path_required",
                score=combined,
            )

        threshold = (
            self.config.hysteresis_release if self.state.speaking else self.config.response_threshold
        )
        if signal.should_respond and combined >= threshold and not cooling_down:
            self.state.last_response_at = now
            self.state.speaking = True
            return ControllerDecision(
                response_action=ResponseAction.RESPOND,
                reasoning_action=ReasoningAction.S1_ONLY,
                reason="s1_response_gate_open",
                score=combined,
            )

        self.state.speaking = False
        return ControllerDecision(
            response_action=ResponseAction.WAIT,
            reasoning_action=ReasoningAction.S1_ONLY,
            reason="silence_preferred" if not cooling_down else "cooldown",
            score=combined,
        )
