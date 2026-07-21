from __future__ import annotations

from pathlib import Path

from dualstream_agent.tools.registry import ToolRegistry


def build_default_tools(workspace: str | Path = ".") -> ToolRegistry:
    """Build conservative read-only workspace tools.

    Mutating tools should be registered explicitly by the application or
    benchmark so the live runtime does not gain write access by default.
    """
    root = Path(workspace).resolve()
    registry = ToolRegistry()

    def resolve_inside_workspace(path: str) -> Path:
        target = (root / path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Path escapes workspace")
        return target

    def list_files(path: str = ".") -> list[str]:
        target = resolve_inside_workspace(path)
        if not target.exists():
            return []
        if target.is_file():
            return [str(target.relative_to(root))]
        return [str(item.relative_to(root)) for item in sorted(target.iterdir())]

    def read_text(path: str) -> str:
        return resolve_inside_workspace(path).read_text(encoding="utf-8")

    registry.register(
        "list_files",
        list_files,
        description="List files in a path under the current workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
            "additionalProperties": False,
        },
    )
    registry.register(
        "read_text",
        read_text,
        description="Read a UTF-8 text file under the current workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    return registry
