from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    kind: Literal["mock", "openai_compatible", "transformers"] = "mock"
    model: str = "mock"
    endpoint: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    device_map: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = True
    multimodal: bool = False
    local_files_only: bool = False


class PerceptionConfig(BaseModel):
    buffer_size: int = 8
    change_threshold: float = 0.08
    silence_ceiling_s: float = 10.0
    thumbnail_size: int = 32


class ControllerConfig(BaseModel):
    response_threshold: float = 0.68
    interrupt_threshold: float = 0.90
    confidence_threshold: float = 0.45
    cooldown_s: float = 3.0
    hysteresis_release: float = 0.50


class MemoryConfig(BaseModel):
    path: str = ".dualstream/memory.sqlite3"
    top_k: int = 5


class SkillConfig(BaseModel):
    root: str = "skills"
    top_k: int = 3
    token_budget_chars: int = 6000


class TraceConfig(BaseModel):
    path: str = ".dualstream/traces.jsonl"


class AppConfig(BaseModel):
    s1: BackendConfig = Field(default_factory=BackendConfig)
    s2: BackendConfig = Field(default_factory=BackendConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillConfig = Field(default_factory=SkillConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    system_prompt: str = (
        "You are a real-time multimodal assistant. Be concise, causal, and avoid "
        "speaking unless the current observation is relevant or urgent."
    )


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
