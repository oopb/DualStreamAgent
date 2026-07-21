from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ResponseAction(str, Enum):
    WAIT = "wait"
    RESPOND = "respond"
    INTERRUPT = "interrupt"
    UPDATE = "update"
    ASK_CLARIFICATION = "ask_clarification"


class ReasoningAction(str, Enum):
    S1_ONLY = "s1_only"
    INVOKE_S2 = "invoke_s2"
    RETRIEVE_MEMORY = "retrieve_memory"
    INVOKE_TOOL = "invoke_tool"


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: Any
    source: str = "stream"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChangeScore:
    score: float
    is_novel: bool
    forced: bool = False
    reason: str = ""


@dataclass(slots=True)
class S1Signal:
    summary: str = ""
    event: str = ""
    relevance: float = 0.0
    confidence: float = 0.0
    urgency: float = 0.0
    novelty: float = 0.0
    need_reasoning: bool = False
    need_memory: bool = False
    need_tool: bool = False
    should_respond: bool = False
    response: str = ""
    memory_note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ControllerDecision:
    response_action: ResponseAction
    reasoning_action: ReasoningAction
    reason: str
    score: float = 0.0


@dataclass(slots=True)
class GenerationRequest:
    messages: list[dict[str, Any]]
    images: list[Any] = field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.0
    stop: list[str] | None = None
    response_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    raw: Any = None


@dataclass(slots=True)
class RuntimeOutput:
    timestamp: float
    decision: ControllerDecision
    signal: S1Signal
    response: str = ""
    trace_id: str = ""


@dataclass(slots=True)
class ScenarioEvent:
    at: float
    kind: str
    payload: dict[str, Any]


@dataclass(slots=True)
class Scenario:
    name: str
    root: Path
    events: list[ScenarioEvent]
    expected: dict[str, Any] = field(default_factory=dict)
