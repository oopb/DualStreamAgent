from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    version: int = 1
    status: SkillStatus = SkillStatus.ACTIVE
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return f"# Skill: {self.name}\n{self.description}\n\n{self.body}".strip()


@dataclass(slots=True)
class SkillRetrievalRecord:
    query: str
    skill_names: list[str]
    top_k: int
    char_budget: int
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SkillManager:
    """Manage isolated candidate and active SKILL.md lifecycles."""

    def __init__(self, root: str | Path = "skills", *, retrieval_log_size: int = 200):
        if retrieval_log_size < 1:
            raise ValueError("retrieval_log_size must be positive")
        self.root = Path(root)
        self.candidates_root = self.root / ".candidates"
        self.root.mkdir(parents=True, exist_ok=True)
        self._retrievals: deque[SkillRetrievalRecord] = deque(maxlen=retrieval_log_size)

    def list_skills(
        self,
        status: SkillStatus | str | None = SkillStatus.ACTIVE,
    ) -> list[Skill]:
        wanted = SkillStatus(status) if status is not None else None
        paths = list(self.root.glob("*/SKILL.md"))
        if wanted in {None, SkillStatus.CANDIDATE}:
            paths.extend(self.candidates_root.glob("*/SKILL.md"))
        skills: list[Skill] = []
        for path in sorted(paths):
            try:
                skill = self._load(path)
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            if wanted is None or skill.status == wanted:
                skills.append(skill)
        return skills

    def read_skill(
        self,
        name: str,
        *,
        status: SkillStatus | str | None = None,
    ) -> Skill:
        safe_name = self._safe_name(name)
        wanted = SkillStatus(status) if status is not None else None
        if wanted == SkillStatus.CANDIDATE:
            paths = [self._candidate_path(safe_name)]
        else:
            paths = [self._active_path(safe_name)]
            if wanted is None:
                paths.append(self._candidate_path(safe_name))
        for path in paths:
            if not path.is_file():
                continue
            skill = self._load(path)
            if wanted is None or skill.status == wanted:
                return skill
        suffix = f" with status={wanted.value}" if wanted is not None else ""
        raise KeyError(f"Unknown skill: {safe_name}{suffix}")

    def retrieve_skills(
        self,
        query: str,
        top_k: int = 3,
        *,
        char_budget: int = 6000,
    ) -> list[Skill]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if char_budget < 1:
            raise ValueError("char_budget must be positive")
        terms = {term.lower() for term in re.findall(r"[\w-]+", query) if len(term) > 1}
        scored: list[tuple[float, Skill]] = []
        for skill in self.list_skills(SkillStatus.ACTIVE):
            title = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            body = skill.body.lower()
            title_hits = sum(2.0 for term in terms if term in title)
            body_hits = sum(0.5 for term in terms if term in body)
            score = title_hits + body_hits
            if not terms or score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda pair: (pair[0], pair[1].version), reverse=True)

        selected: list[Skill] = []
        used = 0
        for _, skill in scored:
            if len(selected) >= top_k or used >= char_budget:
                break
            selected.append(skill)
            used += min(len(skill.render()), char_budget - used)
        self._retrievals.append(
            SkillRetrievalRecord(
                query=query,
                skill_names=[skill.name for skill in selected],
                top_k=top_k,
                char_budget=char_budget,
            )
        )
        return selected

    def recent_retrievals(self, limit: int = 20) -> list[SkillRetrievalRecord]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        records = list(self._retrievals)
        return records[-limit:] if limit else []

    def render_context(self, skills: list[Skill], char_budget: int = 6000) -> str:
        if char_budget < 1:
            return ""
        chunks: list[str] = []
        used = 0
        for skill in skills:
            rendered = skill.render()
            remaining = char_budget - used
            if remaining <= 0:
                break
            chunks.append(rendered[:remaining])
            used += min(len(rendered), remaining)
        return "\n\n---\n\n".join(chunks)

    def write_candidate_skill(
        self,
        *,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        safe_name = self._safe_name(name)
        if not description.strip():
            raise ValueError("Skill description must not be empty")
        if not body.strip():
            raise ValueError("Skill body must not be empty")
        try:
            active_version = self.read_skill(
                safe_name,
                status=SkillStatus.ACTIVE,
            ).version
        except KeyError:
            try:
                active_version = self.read_skill(
                    safe_name,
                    status=SkillStatus.DISABLED,
                ).version
            except KeyError:
                active_version = 0
        frontmatter = {
            **(metadata or {}),
            "name": safe_name,
            "description": description.strip(),
            "version": active_version + 1,
            "status": SkillStatus.CANDIDATE.value,
            "tags": [str(tag) for tag in tags or []],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._candidate_path(safe_name)
        self._write(path, frontmatter, body)
        return path

    def promote_skill(self, name: str, validation: dict[str, Any]) -> Path:
        if validation.get("promoted") is not True:
            raise ValueError("Candidate promotion requires a successful validation result")
        candidate = self.read_skill(name, status=SkillStatus.CANDIDATE)
        metadata = {
            **candidate.metadata,
            "status": SkillStatus.ACTIVE.value,
            "validation": dict(validation),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._active_path(candidate.name)
        self._write(path, metadata, candidate.body)
        candidate.path.unlink()
        candidate.path.parent.rmdir()
        return path

    def record_candidate_validation(
        self,
        name: str,
        validation: dict[str, Any],
    ) -> Path:
        candidate = self.read_skill(name, status=SkillStatus.CANDIDATE)
        metadata = {
            **candidate.metadata,
            "validation": dict(validation),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(candidate.path, metadata, candidate.body)
        return candidate.path

    def disable_skill(self, name: str, *, reason: str = "") -> Path:
        skill = self.read_skill(name, status=SkillStatus.ACTIVE)
        metadata = {
            **skill.metadata,
            "status": SkillStatus.DISABLED.value,
            "disabled_at": datetime.now(timezone.utc).isoformat(),
            "disabled_reason": reason.strip(),
        }
        self._write(skill.path, metadata, skill.body)
        return skill.path

    # Compatibility aliases for the original MVP API.
    def retrieve(self, query: str, top_k: int = 3) -> list[Skill]:
        return self.retrieve_skills(query, top_k=top_k)

    def write_candidate(self, **kwargs: Any) -> Path:
        return self.write_candidate_skill(**kwargs)

    def promote(self, name: str, validation: dict[str, Any]) -> Path:
        return self.promote_skill(name, validation)

    def _active_path(self, name: str) -> Path:
        return self.root / name / "SKILL.md"

    def _candidate_path(self, name: str) -> Path:
        return self.candidates_root / name / "SKILL.md"

    @staticmethod
    def _safe_name(name: str) -> str:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
        if not safe_name:
            raise ValueError("Skill name is empty after sanitization")
        return safe_name

    @staticmethod
    def _write(path: Path, metadata: dict[str, Any], body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            f"---\n{yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)}"
            f"---\n\n{body.strip()}\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _load(path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            metadata: dict[str, Any] = {}
            body = text.strip()
        else:
            parts = text.split("---", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid frontmatter: {path}")
            loaded = yaml.safe_load(parts[1]) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Skill frontmatter must be an object: {path}")
            metadata = loaded
            body = parts[2].strip()
        version = int(metadata.get("version", 1))
        if version < 1:
            raise ValueError(f"Skill version must be positive: {path}")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(f"Skill tags must be a list: {path}")
        return Skill(
            name=str(metadata.get("name") or path.parent.name),
            description=str(metadata.get("description") or ""),
            body=body,
            path=path,
            version=version,
            status=SkillStatus(str(metadata.get("status", SkillStatus.ACTIVE.value))),
            tags=[str(tag) for tag in tags],
            metadata=metadata,
        )
