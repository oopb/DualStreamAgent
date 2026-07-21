from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PerceptionState:
    current_summary: str = ""
    current_event: str = ""
    last_timestamp: float = 0.0
    history: list[str] = field(default_factory=list)

    def update(self, summary: str, event: str, timestamp: float, history_limit: int = 12) -> None:
        self.current_summary = summary
        self.current_event = event
        self.last_timestamp = timestamp
        if summary:
            self.history.append(f"[{timestamp:.2f}] {summary}")
            del self.history[:-history_limit]

    def render(self) -> str:
        if not self.history:
            return "No prior observations."
        return "\n".join(self.history)
