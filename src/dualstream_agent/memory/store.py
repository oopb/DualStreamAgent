from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from dualstream_agent.utils import ensure_parent


@dataclass(slots=True)
class MemoryItem:
    content: str
    kind: str = "episodic"
    summary: str = ""
    importance: float = 0.5
    confidence: float = 1.0
    session_id: str = ""
    timestamp: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MemoryStore:
    """Small persistent SQLite memory store with lexical retrieval."""

    def __init__(self, path: str = ".dualstream/memory.sqlite3"):
        self.path = ensure_parent(path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                session_id TEXT NOT NULL,
                event_timestamp REAL,
                tags_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def add(self, item: MemoryItem) -> str:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, kind, content, summary, importance, confidence,
                    session_id, event_timestamp, tags_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.kind,
                    item.content,
                    item.summary,
                    item.importance,
                    item.confidence,
                    item.session_id,
                    item.timestamp,
                    json.dumps(item.tags, ensure_ascii=False),
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.created_at,
                ),
            )
            self._connection.commit()
        return item.id

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        kinds: set[str] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(sorted(kinds))
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()

        terms = {term.lower() for term in query.split() if len(term) > 1}
        scored: list[tuple[float, MemoryItem]] = []
        for row in rows:
            item = self._from_row(row)
            haystack = f"{item.summary} {item.content} {' '.join(item.tags)}".lower()
            overlap = sum(1.0 for term in terms if term in haystack)
            recency_bonus = 0.1
            score = overlap + item.importance * 0.25 + item.confidence * 0.1 + recency_bonus
            if not terms or overlap > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def recent(self, limit: int = 20, session_id: str | None = None) -> list[MemoryItem]:
        if session_id:
            query = "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at DESC LIMIT ?"
            params = (session_id, limit)
        else:
            query = "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            summary=row["summary"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            session_id=row["session_id"],
            timestamp=row["event_timestamp"],
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def export(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.recent(limit=100000)]
