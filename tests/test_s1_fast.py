import json

import pytest

from dualstream_agent.backends.mock import MockBackend
from dualstream_agent.systems.s1_fast import S1FastSystem


@pytest.mark.asyncio
async def test_s1_uses_safe_defaults_for_malformed_output():
    system = S1FastSystem(MockBackend(["not json"]), "system")

    signal = await system.observe(
        image=None,
        user_goal="watch",
        state_context="none",
        change_score=0.4,
    )

    assert signal.should_respond is False
    assert signal.need_reasoning is False
    assert signal.novelty == pytest.approx(0.4)
    assert signal.semantic_change == 0.0
    assert signal.raw["model_output"] == "not json"


@pytest.mark.asyncio
async def test_s1_does_not_treat_false_strings_as_true():
    response = json.dumps(
        {
            "semantic_change": 2.0,
            "need_reasoning": "false",
            "need_memory": "no",
            "need_tool": "0",
            "should_respond": "false",
            "reason": "nothing changed",
        }
    )
    system = S1FastSystem(MockBackend([response]), "system")

    signal = await system.observe(
        image=None,
        user_goal="watch",
        state_context="none",
        change_score=0.4,
    )

    assert signal.need_reasoning is False
    assert signal.need_memory is False
    assert signal.need_tool is False
    assert signal.should_respond is False
    assert signal.semantic_change == 1.0
    assert signal.reason == "nothing changed"
