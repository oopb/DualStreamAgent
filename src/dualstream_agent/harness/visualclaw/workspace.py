from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import AppliedUpdate, KeyframeSet, VisualClawScenario


@dataclass(slots=True)
class VisualClawWorkCopy:
    run_dir: Path
    workspace: Path
    sessions: Path
    agent_logs: Path
    access_logs: Path
    clip_frames: Path
    scenario: VisualClawScenario


def _resolve_inside(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes staged root: {relative}")
    return target


def stage_work_copy(scenario: VisualClawScenario, run_dir: str | Path) -> VisualClawWorkCopy:
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    workspace = run_path / "workspace"
    sessions = run_path / "sessions"
    for target in (workspace, sessions):
        if target.exists():
            shutil.rmtree(target)
    if scenario.workspace_dir.exists():
        shutil.copytree(scenario.workspace_dir, workspace)
    else:
        workspace.mkdir(parents=True)
    if scenario.sessions_dir.exists():
        shutil.copytree(scenario.sessions_dir, sessions)
    else:
        sessions.mkdir(parents=True)
    agent_logs = run_path / "agent_logs"
    access_logs = run_path / "access_logs"
    clip_frames = workspace / "_clip_frames"
    for directory in (agent_logs, access_logs, clip_frames):
        directory.mkdir(parents=True, exist_ok=True)
    return VisualClawWorkCopy(
        run_dir=run_path,
        workspace=workspace,
        sessions=sessions,
        agent_logs=agent_logs,
        access_logs=access_logs,
        clip_frames=clip_frames,
        scenario=scenario,
    )


def stage_keyframes(work_copy: VisualClawWorkCopy, keyframes: KeyframeSet) -> list[Path]:
    paths: list[Path] = []
    for label, image in zip(keyframes.labels, keyframes.images):
        filename = label.replace(":", "_") + ".jpg"
        target = _resolve_inside(work_copy.clip_frames, filename)
        target.write_bytes(image)
        paths.append(target)
    return paths


def _scope_root(work_copy: VisualClawWorkCopy, action: dict[str, Any], target: str) -> tuple[str, Path, str]:
    scope = str(action.get("scope") or action.get("root") or "workspace").lower()
    normalized = target.replace("\\", "/").lstrip("/")
    if normalized.startswith("workspace/"):
        return "workspace", work_copy.workspace, normalized[len("workspace/") :]
    if normalized.startswith("sessions/"):
        return "sessions", work_copy.sessions, normalized[len("sessions/") :]
    if scope in {"session", "sessions"}:
        return "sessions", work_copy.sessions, normalized
    return "workspace", work_copy.workspace, normalized


def _apply_action(
    work_copy: VisualClawWorkCopy,
    update_id: str,
    update_dir: Path,
    action: dict[str, Any],
) -> AppliedUpdate | None:
    verb = str(action.get("action") or action.get("op") or "new").lower()
    target_value = action.get("target") or action.get("path") or action.get("destination")
    if not target_value:
        return None
    scope, root, relative = _scope_root(work_copy, action, str(target_value))
    target = _resolve_inside(root, relative)
    source_value = action.get("source") or action.get("src")
    source = _resolve_inside(update_dir, str(source_value)) if source_value else None

    if verb in {"new", "replace", "copy", "overwrite", "write"}:
        if source is None or not source.exists():
            raise FileNotFoundError(f"Update {update_id} source missing: {source_value}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists() and verb in {"replace", "overwrite"}:
                shutil.rmtree(target)
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    elif verb == "append":
        target.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            with target.open("ab") as handle:
                handle.write(source.read_bytes())
        else:
            text = str(action.get("content") or action.get("text") or "")
            with target.open("a", encoding="utf-8") as handle:
                handle.write(text)
    elif verb == "delete":
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    elif verb in {"mkdir", "directory"}:
        target.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Unsupported VisualClawArena update action: {verb}")

    return AppliedUpdate(
        update_id=update_id,
        action=verb,
        target=str(target.relative_to(root)),
        source=str(source.relative_to(update_dir)) if source is not None else None,
        scope=scope,
    )


def _manifest_actions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for key in ("files", "actions", "updates", "session_appends", "sessions"):
        value = manifest.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            if key in {"session_appends", "sessions"}:
                copied.setdefault("scope", "sessions")
                copied.setdefault("action", "append")
            actions.append(copied)
    return actions


def apply_update(work_copy: VisualClawWorkCopy, update_id: str) -> list[AppliedUpdate]:
    update_dir = work_copy.scenario.updates_dir / update_id
    if not update_dir.is_dir():
        raise FileNotFoundError(f"VisualClawArena update not found: {update_dir}")
    manifest_path = update_dir / "update_manifest.json"
    applied: list[AppliedUpdate] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Update manifest must be an object: {manifest_path}")
        for action in _manifest_actions(manifest):
            record = _apply_action(work_copy, update_id, update_dir, action)
            if record is not None:
                applied.append(record)
        if applied:
            return applied

    for source in sorted(update_dir.rglob("*")):
        if not source.is_file() or source.name == "update_manifest.json":
            continue
        relative = source.relative_to(update_dir)
        if relative.parts and relative.parts[0] == "sessions":
            root = work_copy.sessions
            target_relative = Path(*relative.parts[1:])
            scope = "sessions"
        else:
            root = work_copy.workspace
            target_relative = relative
            scope = "workspace"
        target = _resolve_inside(root, target_relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        applied.append(
            AppliedUpdate(
                update_id=update_id,
                action="overlay",
                target=str(target_relative),
                source=str(relative),
                scope=scope,
            )
        )
    return applied


def apply_updates(work_copy: VisualClawWorkCopy, update_ids: list[str]) -> list[AppliedUpdate]:
    records: list[AppliedUpdate] = []
    for update_id in update_ids:
        records.extend(apply_update(work_copy, update_id))
    return records
