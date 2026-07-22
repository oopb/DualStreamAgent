from __future__ import annotations

import time
import uuid
from typing import Any

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.backends.factory import build_backend
from dualstream_agent.config import AppConfig
from dualstream_agent.memory.store import MemoryItem, MemoryStore
from dualstream_agent.perception.change_detector import ChangeDetector
from dualstream_agent.perception.frame_buffer import FrameBuffer
from dualstream_agent.perception.state import PerceptionState
from dualstream_agent.runtime.trace import TraceStore
from dualstream_agent.schemas import (
    ControllerDecision,
    FramePacket,
    MemoryAction,
    ReasoningAction,
    ResponseAction,
    RuntimeOutput,
    S1Signal,
)
from dualstream_agent.skills.manager import SkillManager
from dualstream_agent.systems.controller import DualController
from dualstream_agent.systems.s1_fast import S1FastSystem
from dualstream_agent.systems.s2_slow import S2SlowSystem
from dualstream_agent.tools.builtins import build_default_tools
from dualstream_agent.tools.registry import ToolRegistry


class DualStreamSession:
    """Single causal session shared by live runtime and benchmark harness."""

    def __init__(
        self,
        config: AppConfig,
        *,
        session_id: str | None = None,
        s1_backend: ModelBackend | None = None,
        s2_backend: ModelBackend | None = None,
        memory: MemoryStore | None = None,
        skills: SkillManager | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.config = config
        self.session_id = session_id or uuid.uuid4().hex
        self.s1_backend = s1_backend or build_backend(config.s1)
        if s2_backend is not None:
            self.s2_backend = s2_backend
        elif s1_backend is None and config.s1.model_dump() == config.s2.model_dump():
            # Avoid loading the same local Transformers model twice. S1 and S2
            # can share weights while using different prompts and token budgets.
            self.s2_backend = self.s1_backend
        else:
            self.s2_backend = build_backend(config.s2)
        self.memory = memory or MemoryStore(config.memory.path)
        self.skills = skills or SkillManager(config.skills.root)
        self.tools = tools or build_default_tools()
        self.trace = TraceStore(config.trace.path)
        self.frames = FrameBuffer(config.perception.buffer_size)
        self.change_detector = ChangeDetector(
            threshold=config.perception.change_threshold,
            silence_ceiling_s=config.perception.silence_ceiling_s,
            thumbnail_size=config.perception.thumbnail_size,
        )
        self.state = PerceptionState()
        self.controller = DualController(config.controller)
        self.s1 = S1FastSystem(self.s1_backend, config.system_prompt)
        self.s2 = S2SlowSystem(
            self.s2_backend,
            self.memory,
            self.skills,
            self.tools,
            memory_top_k=config.memory.top_k,
            skill_top_k=config.skills.top_k,
            skill_char_budget=config.skills.token_budget_chars,
        )
        self.user_goal = ""
        self._frame_id = 0

    def set_user_goal(self, goal: str) -> None:
        self.user_goal = goal.strip()

    async def process_frame(
        self,
        image: Any,
        *,
        timestamp: float | None = None,
        user_goal: str | None = None,
        source: str = "stream",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeOutput:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if user_goal is not None:
            self.set_user_goal(user_goal)
        self._frame_id += 1
        frame = FramePacket(
            frame_id=self._frame_id,
            timestamp=now,
            image=image,
            source=source,
            metadata=metadata or {},
        )
        change = self.change_detector.score(image, now)
        self.frames.append(frame)

        if not change.is_novel:
            decision = ControllerDecision(
                response_action=ResponseAction.WAIT,
                reasoning_action=ReasoningAction.S1_ONLY,
                memory_action=MemoryAction.NONE,
                priority=change.score,
                reason="perception_change_gate_skip",
            )
            trace_id = self.trace.append(
                {
                    "session_id": self.session_id,
                    "timestamp": now,
                    "kind": "frame_skipped",
                    "frame_id": frame.frame_id,
                    "change": change,
                    "decision": decision,
                }
            )
            return RuntimeOutput(
                timestamp=now,
                decision=decision,
                signal=S1Signal(novelty=change.score),
                trace_id=trace_id,
            )

        signal = await self.s1.observe(
            image=image,
            user_goal=self.user_goal,
            state_context=self.state.render(),
            change_score=change.score,
        )
        self.state.update(signal.summary, signal.event, now)
        decision = self.controller.decide(signal, now)
        response = ""

        if decision.reasoning_action in {
            ReasoningAction.INVOKE_S2,
            ReasoningAction.RETRIEVE_MEMORY,
            ReasoningAction.INVOKE_TOOL,
        }:
            response = await self.s2.reason(
                user_goal=self.user_goal,
                signal=signal,
                state_context=self.state.render(),
                image=image,
            )
            if response:
                decision = ControllerDecision(
                    response_action=ResponseAction.RESPOND,
                    reasoning_action=decision.reasoning_action,
                    memory_action=decision.memory_action,
                    priority=decision.priority,
                    reason=f"{decision.reason}_completed",
                )
                self.controller.state.last_response_at = now
        elif decision.response_action in {ResponseAction.RESPOND, ResponseAction.INTERRUPT}:
            response = signal.response or signal.summary

        self._store_episode(
            signal,
            decision,
            now,
            source=source,
            metadata={"frame_id": frame.frame_id},
        )

        trace_id = self.trace.append(
            {
                "session_id": self.session_id,
                "timestamp": now,
                "kind": "frame_processed",
                "frame_id": frame.frame_id,
                "change": change,
                "signal": signal,
                "decision": decision,
                "response": response,
            }
        )
        return RuntimeOutput(
            timestamp=now,
            decision=decision,
            signal=signal,
            response=response,
            trace_id=trace_id,
        )

    async def process_text(self, text: str, *, timestamp: float | None = None) -> RuntimeOutput:
        """Process a user request without requiring a new frame."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        self.set_user_goal(text)
        latest = self.frames.latest()
        signal = await self.s1.observe(
            image=latest.image if latest is not None else None,
            user_goal=text,
            state_context=self.state.render(),
            change_score=1.0,
        )
        decision = self.controller.decide(signal, now)
        response = ""
        if decision.reasoning_action != ReasoningAction.S1_ONLY:
            response = await self.s2.reason(
                user_goal=text,
                signal=signal,
                state_context=self.state.render(),
                image=latest.image if latest is not None else None,
            )
            decision = ControllerDecision(
                response_action=ResponseAction.RESPOND,
                reasoning_action=decision.reasoning_action,
                memory_action=decision.memory_action,
                priority=decision.priority,
                reason=f"{decision.reason}_completed",
            )
        elif decision.response_action in {ResponseAction.RESPOND, ResponseAction.INTERRUPT}:
            response = signal.response or signal.summary
        self._store_episode(signal, decision, now, source="text")
        trace_id = self.trace.append(
            {
                "session_id": self.session_id,
                "timestamp": now,
                "kind": "text_processed",
                "text": text,
                "signal": signal,
                "decision": decision,
                "response": response,
            }
        )
        return RuntimeOutput(now, decision, signal, response, trace_id)

    def _store_episode(
        self,
        signal: S1Signal,
        decision: ControllerDecision,
        timestamp: float,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if decision.memory_action != MemoryAction.STORE_EPISODE:
            return
        content = signal.memory_note or signal.summary
        if not content:
            return
        self.memory.add(
            MemoryItem(
                content=content,
                summary=signal.summary,
                kind="episodic",
                importance=max(signal.relevance, signal.urgency),
                confidence=signal.confidence,
                session_id=self.session_id,
                timestamp=timestamp,
                tags=[signal.event] if signal.event else [],
                metadata={
                    **(metadata or {}),
                    "source": source,
                    "decision": decision.reason,
                },
            )
        )

    async def close(self) -> None:
        if self.s2_backend is not self.s1_backend:
            await self.s2_backend.close()
        await self.s1_backend.close()
        self.memory.close()
