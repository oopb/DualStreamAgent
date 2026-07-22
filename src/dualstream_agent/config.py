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
    buffer_size: int = Field(default=8, ge=1)
    change_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    silence_ceiling_s: float = Field(default=10.0, gt=0.0)
    thumbnail_size: int = Field(default=32, ge=1)


class ControllerConfig(BaseModel):
    response_threshold: float = Field(default=0.68, ge=0.0, le=1.0)
    interrupt_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    cooldown_s: float = Field(default=3.0, ge=0.0)
    hysteresis_release: float = Field(default=0.50, ge=0.0, le=1.0)


class MemoryConfig(BaseModel):
    path: str = ".dualstream/memory.sqlite3"
    top_k: int = Field(default=5, ge=1)


class SkillConfig(BaseModel):
    root: str = "skills"
    top_k: int = Field(default=3, ge=1)
    token_budget_chars: int = Field(default=6000, ge=1)


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
