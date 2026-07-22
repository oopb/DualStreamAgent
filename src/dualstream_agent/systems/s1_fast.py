from __future__ import annotations

from typing import Any

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.schemas import GenerationRequest, S1Signal
from dualstream_agent.utils import clamp01, coerce_bool, extract_json_object

S1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "event": {"type": "string"},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "urgency": {"type": "number", "minimum": 0, "maximum": 1},
        "novelty": {"type": "number", "minimum": 0, "maximum": 1},
        "need_reasoning": {"type": "boolean"},
        "need_memory": {"type": "boolean"},
        "need_tool": {"type": "boolean"},
        "should_respond": {"type": "boolean"},
        "response": {"type": "string"},
        "memory_note": {"type": "string"},
    },
    "required": [
        "summary",
        "event",
        "relevance",
        "confidence",
        "urgency",
        "novelty",
        "need_reasoning",
        "need_memory",
        "need_tool",
        "should_respond",
        "response",
        "memory_note",
    ],
    "additionalProperties": False,
}


class S1FastSystem:
    def __init__(self, backend: ModelBackend, system_prompt: str):
        self.backend = backend
        self.system_prompt = system_prompt

    async def observe(
        self,
        *,
        image: Any | None,
        user_goal: str,
        state_context: str,
        change_score: float,
    ) -> S1Signal:
        prompt = f"""
You are System 1, the fast causal controller of a streaming assistant.
Inspect only the current observation and the supplied past state. Decide both:
1. whether the assistant should speak now; and
2. whether slower System 2 reasoning is required.

Rules:
- Prefer silence when nothing task-relevant changed.
- Urgent safety-relevant observations may trigger an immediate short response.
- Use need_reasoning for long-term recall, ambiguity, planning, verification, or tools.
- Do not claim events that are not visible or present in the causal state.
- Return one JSON object matching the required schema, with no prose outside JSON.

User goal or latest request:
{user_goal or 'No explicit request. Observe passively.'}

Recent causal state:
{state_context}

Cheap visual change score: {change_score:.4f}
""".strip()
        result = await self.backend.generate(
            GenerationRequest(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                images=[image] if image is not None else [],
                max_new_tokens=256,
                temperature=0.0,
                response_schema=S1_SCHEMA,
            )
        )
        data = extract_json_object(result.text)
        return S1Signal(
            summary=str(data.get("summary", "")),
            event=str(data.get("event", "")),
            relevance=clamp01(data.get("relevance")),
            confidence=clamp01(data.get("confidence")),
            urgency=clamp01(data.get("urgency")),
            novelty=clamp01(data.get("novelty"), change_score),
            need_reasoning=coerce_bool(data.get("need_reasoning")),
            need_memory=coerce_bool(data.get("need_memory")),
            need_tool=coerce_bool(data.get("need_tool")),
            should_respond=coerce_bool(data.get("should_respond")),
            response=str(data.get("response", "")),
            memory_note=str(data.get("memory_note", "")),
            raw={"model_output": result.text, "generation": result.raw},
        )
