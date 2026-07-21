from __future__ import annotations

import re
from pathlib import Path

from .schemas import VisualClawRound

SYSTEM_PROMPT = """You are the evaluated agent in a VisualClawArena scenario.

You receive only the current round, the current persistent workspace, already-triggered
updates, session history, and supplied walkthrough frames. Never assume access to
future rounds, checker scripts, gold files, or construction notes.

Evidence citation formats:
- video: [clip @ MM:SS]
- document: [doc:relative/path]
- image: [image:relative/path]
- PDF: [pdf:relative/path]
- audio: [audio:relative/path @ MM:SS]
- chat: [chat:speaker @ HH:MM]

For a multi_choice round, finish with \\bbox{X} on its own line.
For an exec_check round, create or replace requested workspace files by emitting one
or more blocks exactly in this form:

### WRITE_FILE: relative/path/in/workspace.ext
```text
file contents
```

Do not write outside the workspace. The runner applies these blocks before executing
the benchmark checker. Be concise but include enough reasoning to audit the answer.
"""

_TEXT_SUFFIXES = {
    ".md", ".txt", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".py",
    ".toml", ".xml", ".html", ".js", ".ts",
}


def _visible_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and "_clip_frames" not in path.relative_to(root).parts
    )


def workspace_summary(root: Path, *, char_budget: int = 20_000) -> str:
    files = _visible_files(root)
    if not files:
        return "(workspace is empty)"
    chunks: list[str] = []
    used = 0
    for path in files:
        relative = path.relative_to(root)
        remaining = char_budget - used
        if remaining <= 256:
            chunks.append("(workspace summary truncated: character budget exhausted)")
            break
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            chunks.append(f"### {relative}\n(binary/non-text, {path.stat().st_size} bytes)")
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            chunks.append(f"### {relative}\n(unreadable text, {path.stat().st_size} bytes)")
            continue
        body = body[: max(0, remaining - 100)]
        chunks.append(f"### {relative}\n```\n{body}\n```")
        used += len(body) + 100
    return "\n\n".join(chunks)


def sessions_summary(root: Path, *, char_budget: int = 10_000) -> str:
    if not root.exists():
        return "(no session history)"
    chunks: list[str] = []
    used = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        remaining = char_budget - used
        if remaining <= 256:
            break
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        body = body[: max(0, remaining - 100)]
        chunks.append(f"### session: {path.relative_to(root)}\n```\n{body}\n```")
        used += len(body) + 100
    return "\n\n".join(chunks) if chunks else "(no readable session history)"


def build_round_prompt(
    round_item: VisualClawRound,
    *,
    workspace: Path,
    sessions: Path,
    keyframe_labels: list[str],
    workspace_char_budget: int = 20_000,
    sessions_char_budget: int = 10_000,
) -> str:
    parts = [
        f"# Round {round_item.round_number}: {round_item.round_id}",
        f"## Type\n{round_item.round_type}",
        f"## Current task\n{round_item.question}",
    ]
    if round_item.round_type == "multi_choice":
        options = round_item.evaluation.get("options") or {}
        if isinstance(options, dict) and options:
            parts.append(
                "## Options\n"
                + "\n".join(f"- **{key}**: {value}" for key, value in options.items())
            )
        elif isinstance(options, list) and options:
            parts.append(
                "## Options\n"
                + "\n".join(f"- **{chr(65 + index)}**: {value}" for index, value in enumerate(options))
            )
        parts.append("End the answer with `\\bbox{X}` on its own line.")
    else:
        parts.append(
            "This is an exec_check round. Emit actual `### WRITE_FILE:` blocks for every "
            "requested file. Printing a description without a write block will not modify the workspace."
        )
    if keyframe_labels:
        parts.append(
            "## Supplied walkthrough frames\n"
            + ", ".join(f"[clip @ {label}]" for label in keyframe_labels)
            + "\nThe image attachments appear in the same order as these timestamps."
        )
    else:
        parts.append("## Supplied walkthrough frames\n(none)")
    parts.append(
        "## Current persistent workspace\n"
        + workspace_summary(workspace, char_budget=workspace_char_budget)
    )
    parts.append(
        "## Session history visible so far\n"
        + sessions_summary(sessions, char_budget=sessions_char_budget)
    )
    return "\n\n".join(parts)


_WRITE_FILE_RE = re.compile(
    r"###\s*WRITE_FILE:\s*([^\r\n]+?)\s*\r?\n```[^\r\n]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def apply_write_blocks(response: str, workspace: Path) -> list[str]:
    workspace = workspace.resolve()
    written: list[str] = []
    for match in _WRITE_FILE_RE.finditer(response or ""):
        relative_text = match.group(1).strip().replace("\\", "/").lstrip("/")
        relative = Path(relative_text)
        if not relative_text or ".." in relative.parts:
            continue
        target = (workspace / relative).resolve()
        if target != workspace and workspace not in target.parents:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(match.group(2), encoding="utf-8")
        written.append(str(target.relative_to(workspace)))
    return written
