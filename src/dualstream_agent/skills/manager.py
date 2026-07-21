from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    version: int = 1
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return f"# Skill: {self.name}\n{self.description}\n\n{self.body}".strip()


class SkillManager:
    def __init__(self, root: str = "skills"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_skills(self, status: str | None = "active") -> list[Skill]:
        skills: list[Skill] = []
        for path in sorted(self.root.glob("*/SKILL.md")):
            try:
                skill = self._load(path)
            except (OSError, ValueError, yaml.YAMLError):
                continue
            if status is None or skill.status == status:
                skills.append(skill)
        return skills

    def read_skill(self, name: str) -> Skill:
        path = self.root / name / "SKILL.md"
        if not path.exists():
            raise KeyError(f"Unknown skill: {name}")
        return self._load(path)

    def retrieve(self, query: str, top_k: int = 3) -> list[Skill]:
        terms = {term.lower() for term in re.findall(r"[\w-]+", query) if len(term) > 1}
        scored: list[tuple[float, Skill]] = []
        for skill in self.list_skills():
            title = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            body = skill.body.lower()
            title_hits = sum(2.0 for term in terms if term in title)
            body_hits = sum(0.5 for term in terms if term in body)
            score = title_hits + body_hits
            if not terms or score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda pair: (pair[0], pair[1].version), reverse=True)
        return [skill for _, skill in scored[:top_k]]

    def render_context(self, skills: list[Skill], char_budget: int = 6000) -> str:
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

    def write_candidate(
        self,
        *,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
        if not safe_name:
            raise ValueError("Skill name is empty after sanitization")
        path = self.root / safe_name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = {
            "name": safe_name,
            "description": description.strip(),
            "version": 1,
            "status": "candidate",
            "tags": tags or [],
            **(metadata or {}),
        }
        path.write_text(
            f"---\n{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)}---\n\n{body.strip()}\n",
            encoding="utf-8",
        )
        return path

    def promote(self, name: str, validation: dict[str, Any]) -> Path:
        skill = self.read_skill(name)
        metadata = dict(skill.metadata)
        metadata["status"] = "active"
        metadata["version"] = skill.version + 1
        metadata["validation"] = validation
        path = skill.path
        path.write_text(
            f"---\n{yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)}---\n\n{skill.body.strip()}\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _load(path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            metadata: dict[str, Any] = {}
            body = text
        else:
            parts = text.split("---", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid frontmatter: {path}")
            metadata = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
        name = str(metadata.get("name") or path.parent.name)
        description = str(metadata.get("description") or "")
        return Skill(
            name=name,
            description=description,
            body=body,
            path=path,
            version=int(metadata.get("version", 1)),
            status=str(metadata.get("status", "active")),
            tags=list(metadata.get("tags", [])),
            metadata=metadata,
        )
