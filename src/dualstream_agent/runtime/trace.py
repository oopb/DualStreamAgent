from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dualstream_agent.utils import ensure_parent


class TraceStore:
    def __init__(self, path: str = ".dualstream/traces.jsonl"):
        self.path = ensure_parent(path)
        self._lock = threading.RLock()

    def append(self, event: dict[str, Any]) -> str:
        trace_id = str(event.get("trace_id") or uuid.uuid4().hex)
        payload = {
            "trace_id": trace_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=self._json_default) + "\n")
        return trace_id

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    @staticmethod
    def _json_default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "value"):
            return value.value
        return str(value)
