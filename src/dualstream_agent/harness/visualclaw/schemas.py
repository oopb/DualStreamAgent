from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RoundType = Literal["multi_choice", "exec_check"]


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _field(data: dict[str, Any], meta: dict[str, Any], name: str, default: Any = None) -> Any:
    if name in data:
        return data[name]
    return meta.get(name, default)


@dataclass(slots=True)
class VisualClawRound:
    round_id: str
    round_number: int
    question: str
    round_type: RoundType
    evaluation: dict[str, Any] = field(default_factory=dict)
    update_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
    anti_skills: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    video_required: bool = False
    evidence_type: str = ""
    included_in_release_eval: bool = True
    deprecated: bool = False
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int = 0) -> "VisualClawRound":
        meta_value = data.get("meta") or data.get("metadata") or {}
        meta = dict(meta_value) if isinstance(meta_value, dict) else {}
        round_id = str(
            data.get("id")
            or data.get("round_id")
            or data.get("qid")
            or f"round_{index + 1:03d}"
        )
        raw_type = str(data.get("type") or data.get("round_type") or "multi_choice")
        if raw_type not in {"multi_choice", "exec_check"}:
            raise ValueError(f"Unsupported VisualClawArena round type: {raw_type!r}")
        update_ids = (
            data.get("update_ids")
            or data.get("updates")
            or data.get("trigger_updates")
            or []
        )
        round_number = _field(data, meta, "round_number")
        if round_number is None:
            round_number = _field(data, meta, "round", index + 1)
        return cls(
            round_id=round_id,
            round_number=int(round_number),
            question=str(data.get("question") or data.get("instruction") or ""),
            round_type=raw_type,  # type: ignore[arg-type]
            evaluation=dict(data.get("eval") or data.get("evaluation") or {}),
            update_ids=_as_string_list(update_ids),
            expected_sources=_as_string_list(_field(data, meta, "expected_sources")),
            required_modalities=_as_string_list(_field(data, meta, "required_modalities")),
            required_skills=_as_string_list(_field(data, meta, "required_skills")),
            anti_skills=_as_string_list(_field(data, meta, "anti_skills")),
            tags=_as_string_list(_field(data, meta, "tags")),
            video_required=bool(_field(data, meta, "video_required", False)),
            evidence_type=str(_field(data, meta, "evidence_type", "") or ""),
            included_in_release_eval=bool(
                _field(data, meta, "included_in_release_eval", True)
            ),
            deprecated=bool(_field(data, meta, "deprecated", False)),
            feedback=str(data.get("feedback") or ""),
            metadata=meta,
            raw=dict(data),
        )

    def as_official_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload.setdefault("id", self.round_id)
        payload.setdefault("round_id", self.round_id)
        payload.setdefault("round_number", self.round_number)
        payload.setdefault("question", self.question)
        payload.setdefault("type", self.round_type)
        payload.setdefault("eval", self.evaluation)
        payload.setdefault("update_ids", self.update_ids)
        payload.setdefault("meta", self.metadata)
        return payload


@dataclass(slots=True)
class VisualClawScenario:
    scenario_id: str
    dataset_root: Path
    scenario_root: Path
    data_dir: Path
    spec_dir: Path
    workspace_dir: Path
    sessions_dir: Path
    updates_dir: Path
    clip_paths: list[Path]
    rounds: list[VisualClawRound]
    source_bucket: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppliedUpdate:
    update_id: str
    action: str
    target: str
    source: str | None = None
    scope: str = "workspace"


@dataclass(slots=True)
class KeyframeSet:
    images: list[bytes] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    mode: str = "none"
    clip_path: Path | None = None


@dataclass(slots=True)
class AgentTurn:
    response: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    raw: Any = None
