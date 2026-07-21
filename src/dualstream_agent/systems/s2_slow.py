from __future__ import annotations

import json
from typing import Any

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.memory.store import MemoryStore
from dualstream_agent.schemas import GenerationRequest, S1Signal
from dualstream_agent.skills.manager import SkillManager
from dualstream_agent.tools.registry import ToolRegistry
from dualstream_agent.utils import extract_json_object


class S2SlowSystem:
    def __init__(
        self,
        backend: ModelBackend,
        memory: MemoryStore,
        skills: SkillManager,
        tools: ToolRegistry,
        *,
        memory_top_k: int = 5,
        skill_top_k: int = 3,
        skill_char_budget: int = 6000,
        max_tool_steps: int = 3,
    ):
        self.backend = backend
        self.memory = memory
        self.skills = skills
        self.tools = tools
        self.memory_top_k = memory_top_k
        self.skill_top_k = skill_top_k
        self.skill_char_budget = skill_char_budget
        self.max_tool_steps = max_tool_steps

    async def reason(
        self,
        *,
        user_goal: str,
        signal: S1Signal,
        state_context: str,
        image: Any | None = None,
    ) -> str:
        query = " ".join(filter(None, [user_goal, signal.summary, signal.event]))
        memories = self.memory.search(query, top_k=self.memory_top_k)
        skills = self.skills.retrieve(query, top_k=self.skill_top_k)
        memory_context = "\n".join(
            f"- [{item.kind}] {item.summary or item.content}" for item in memories
        ) or "No relevant long-term memory."
        skill_context = self.skills.render_context(skills, self.skill_char_budget) or "No relevant skills."
        tool_schemas = self.tools.schemas()

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are System 2, the slow reasoning component of a real-time assistant. "
                    "Use causal evidence, memory, skills, and tools. Do not invent tool results. "
                    "When a tool is needed, output JSON with action='tool', tool, arguments. "
                    "Otherwise output JSON with action='final' and response."
                ),
            },
            {
                "role": "user",
                "content": f"""
User goal:
{user_goal or 'No explicit request.'}

System 1 observation:
{signal.summary}
Event: {signal.event}

Recent state:
{state_context}

Relevant memory:
{memory_context}

Relevant skills:
{skill_context}

Available tools:
{json.dumps(tool_schemas, ensure_ascii=False)}
""".strip(),
            },
        ]

        for _ in range(self.max_tool_steps + 1):
            result = await self.backend.generate(
                GenerationRequest(
                    messages=messages,
                    images=[image] if image is not None else [],
                    max_new_tokens=768,
                    temperature=0.0,
                )
            )
            data = extract_json_object(result.text)
            if data.get("action") != "tool":
                return str(data.get("response") or result.text).strip()

            tool_name = str(data.get("tool", ""))
            arguments = data.get("arguments", {})
            try:
                tool_result = await self.tools.call(
                    tool_name,
                    arguments if isinstance(arguments, dict) else {},
                )
                rendered = json.dumps(tool_result, ensure_ascii=False, default=str)
            except Exception as exc:  # Tool failures become model-visible evidence.
                rendered = json.dumps({"error": str(exc)}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": result.text})
            messages.append(
                {"role": "user", "content": f"Tool result for {tool_name}: {rendered}"}
            )

        return "System 2 reached the tool-step limit without a final answer."
