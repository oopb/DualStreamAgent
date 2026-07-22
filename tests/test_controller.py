from dualstream_agent.config import ControllerConfig
from dualstream_agent.schemas import MemoryAction, ReasoningAction, ResponseAction, S1Signal
from dualstream_agent.systems.controller import DualController


def test_controller_interrupts_urgent_event():
    controller = DualController(ControllerConfig())
    decision = controller.decide(
        S1Signal(urgency=0.95, relevance=0.8, confidence=0.8, novelty=0.8),
        now=1.0,
    )
    assert decision.response_action == ResponseAction.INTERRUPT
    assert decision.memory_action == MemoryAction.NONE
    assert decision.priority == 0.95


def test_controller_routes_uncertain_signal_to_s2():
    controller = DualController(ControllerConfig())
    decision = controller.decide(
        S1Signal(relevance=0.8, confidence=0.1, novelty=0.8),
        now=1.0,
    )
    assert decision.reasoning_action == ReasoningAction.INVOKE_S2


def test_controller_can_store_memory_while_waiting():
    controller = DualController(ControllerConfig())
    decision = controller.decide(
        S1Signal(
            summary="The cup moved.",
            memory_note="Cup moved near the edge.",
            need_reasoning=True,
            confidence=0.8,
        ),
        now=1.0,
    )

    assert decision.response_action == ResponseAction.WAIT
    assert decision.reasoning_action == ReasoningAction.INVOKE_S2
    assert decision.memory_action == MemoryAction.STORE_EPISODE


def test_controller_applies_cooldown_between_responses():
    controller = DualController(ControllerConfig(cooldown_s=3.0))
    signal = S1Signal(
        relevance=1.0,
        confidence=1.0,
        novelty=1.0,
        should_respond=True,
    )

    assert controller.decide(signal, now=0.0).response_action == ResponseAction.RESPOND
    cooling_down = controller.decide(signal, now=1.0)
    assert cooling_down.response_action == ResponseAction.WAIT
    assert cooling_down.reason == "cooldown"
    assert controller.decide(signal, now=3.0).response_action == ResponseAction.RESPOND
