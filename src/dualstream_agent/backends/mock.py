from __future__ import annotations

import json
from collections import deque

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.schemas import GenerationRequest, GenerationResult


class MockBackend(ModelBackend):
    """Deterministic backend for tests and harness dry-runs."""

    def __init__(self, responses: list[str] | None = None):
        default = json.dumps(
            {
                "summary": "No important change.",
                "event": "none",
                "semantic_change": 0.0,
                "relevance": 0.0,
                "confidence": 1.0,
                "urgency": 0.0,
                "novelty": 0.0,
                "need_reasoning": False,
                "need_memory": False,
                "need_tool": False,
                "should_respond": False,
                "response": "",
                "memory_note": "",
                "reason": "no relevant event",
            }
        )
        self.responses = deque(responses or [default])
        self.default = default

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        text = self.responses.popleft() if self.responses else self.default
        return GenerationResult(text=text, raw={"mock": True})
