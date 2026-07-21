from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from dualstream_agent.schemas import FramePacket


class FrameBuffer:
    def __init__(self, maxlen: int = 8):
        if maxlen < 1:
            raise ValueError("maxlen must be positive")
        self._frames: deque[FramePacket] = deque(maxlen=maxlen)

    def append(self, frame: FramePacket) -> None:
        self._frames.append(frame)

    def latest(self) -> FramePacket | None:
        return self._frames[-1] if self._frames else None

    def recent(self, count: int | None = None) -> list[FramePacket]:
        frames = list(self._frames)
        return frames if count is None else frames[-count:]

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterable[FramePacket]:
        return iter(self._frames)
