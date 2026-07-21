from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class MemoryItem:
    content: str
    kind: str = "episodic"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class MemoryStore:
    def __init__(self):
        self.items = []

    def add(self, item: MemoryItem):
        self.items.append(item)

    def search(self, keyword: str):
        return [asdict(x) for x in self.items if keyword.lower() in x.content.lower()]
