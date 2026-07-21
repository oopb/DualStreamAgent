from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from dualstream_agent.schemas import ChangeScore


def _to_gray_thumbnail(image: Any, size: int) -> np.ndarray:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32) / 255.0


class ChangeDetector:
    """Cheap causal visual novelty filter.

    It compares the current frame with the last accepted frame. The silence
    ceiling forces occasional observations through even in static scenes.
    """

    def __init__(
        self,
        threshold: float = 0.08,
        silence_ceiling_s: float = 10.0,
        thumbnail_size: int = 32,
    ):
        self.threshold = threshold
        self.silence_ceiling_s = silence_ceiling_s
        self.thumbnail_size = thumbnail_size
        self._reference: np.ndarray | None = None
        self._last_accepted_ts: float | None = None

    def score(self, image: Any, timestamp: float) -> ChangeScore:
        current = _to_gray_thumbnail(image, self.thumbnail_size)
        if self._reference is None:
            self._accept(current, timestamp)
            return ChangeScore(score=1.0, is_novel=True, forced=True, reason="first_frame")

        difference = float(np.mean(np.abs(current - self._reference)))
        silence = (
            self._last_accepted_ts is None
            or timestamp - self._last_accepted_ts >= self.silence_ceiling_s
        )
        novel = difference >= self.threshold or silence
        if novel:
            self._accept(current, timestamp)
        return ChangeScore(
            score=difference,
            is_novel=novel,
            forced=silence and difference < self.threshold,
            reason="silence_ceiling" if silence and difference < self.threshold else "visual_change",
        )

    def reset(self) -> None:
        self._reference = None
        self._last_accepted_ts = None

    def _accept(self, current: np.ndarray, timestamp: float) -> None:
        self._reference = current
        self._last_accepted_ts = timestamp
