from __future__ import annotations

from pathlib import Path

import yaml

from dualstream_agent.schemas import Scenario, ScenarioEvent


def load_scenario(path: str | Path) -> Scenario:
    source = Path(path)
    if source.is_dir():
        source = source / "scenario.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    events = [
        ScenarioEvent(
            at=float(item.get("at", 0.0)),
            kind=str(item["kind"]),
            payload=dict(item.get("payload", {})),
        )
        for item in data.get("events", [])
    ]
    events.sort(key=lambda event: event.at)
    return Scenario(
        name=str(data.get("name", source.parent.name)),
        root=source.parent,
        events=events,
        expected=dict(data.get("expected", {})),
    )
