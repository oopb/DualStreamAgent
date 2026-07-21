from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    WAIT = "wait"
    RESPOND = "respond"
    INTERRUPT = "interrupt"
    INVOKE_S2 = "invoke_s2"


@dataclass
class S1Signal:
    relevance: float
    confidence: float
    urgency: float
    need_reasoning: bool = False


class DualController:
    """First version of the dual-system scheduler.

    It separates response timing from slow reasoning invocation.
    """

    def decide(self, signal: S1Signal) -> Action:
        if signal.urgency >= 0.9:
            return Action.INTERRUPT
        if signal.need_reasoning:
            return Action.INVOKE_S2
        if signal.relevance >= 0.7 and signal.confidence >= 0.5:
            return Action.RESPOND
        return Action.WAIT
