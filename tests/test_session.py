import json

import numpy as np
import pytest

from dualstream_agent.backends.mock import MockBackend
from dualstream_agent.config import AppConfig, MemoryConfig, SkillConfig, TraceConfig
from dualstream_agent.runtime.session import DualStreamSession
from dualstream_agent.schemas import ResponseAction


@pytest.mark.asyncio
async def test_session_can_respond_from_s1(tmp_path):
    response = json.dumps(
        {
            "summary": "A cup is falling.",
            "event": "cup_falling",
            "relevance": 1.0,
            "confidence": 1.0,
            "urgency": 0.95,
            "novelty": 1.0,
            "need_reasoning": False,
            "need_memory": False,
            "need_tool": False,
            "should_respond": True,
            "response": "The cup is falling!",
            "memory_note": "The cup began falling.",
        }
    )
    config = AppConfig(
        memory=MemoryConfig(path=str(tmp_path / "memory.sqlite3")),
        skills=SkillConfig(root=str(tmp_path / "skills")),
        trace=TraceConfig(path=str(tmp_path / "traces.jsonl")),
    )
    session = DualStreamSession(
        config,
        s1_backend=MockBackend([response]),
        s2_backend=MockBackend(),
    )
    output = await session.process_frame(np.zeros((16, 16, 3), dtype=np.uint8), timestamp=0.0)
    assert output.decision.response_action == ResponseAction.INTERRUPT
    assert output.response == "The cup is falling!"
    await session.close()
