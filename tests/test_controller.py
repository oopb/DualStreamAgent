from dualstream_agent.config import ControllerConfig
from dualstream_agent.schemas import ReasoningAction, ResponseAction, S1Signal
from dualstream_agent.systems.controller import DualController


def test_controller_interrupts_urgent_event():
    controller = DualController(ControllerConfig())
    decision = controller.decide(
        S1Signal(urgency=0.95, relevance=0.8, confidence=0.8, novelty=0.8),
        now=1.0,
    )
    assert decision.response_action == ResponseAction.INTERRUPT


def test_controller_routes_uncertain_signal_to_s2():
    controller = DualController(ControllerConfig())
    decision = controller.decide(
        S1Signal(relevance=0.8, confidence=0.1, novelty=0.8),
        now=1.0,
    )
    assert decision.reasoning_action == ReasoningAction.INVOKE_S2
