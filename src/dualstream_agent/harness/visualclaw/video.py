from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .schemas import KeyframeSet, VisualClawRound

_CLIP_CITATION_RE = re.compile(r"\[clip\s*@\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\]", re.IGNORECASE)


def parse_cited_timestamps(rounds: Iterable[VisualClawRound]) -> list[float]:
    timestamps: set[float] = set()
    for item in rounds:
        text = f"{item.question} {item.feedback}"
        for match in _CLIP_CITATION_RE.finditer(text):
            first, second, third = match.groups()
            seconds = (
                int(first) * 60 + int(second)
                if third is None
                else int(first) * 3600 + int(second) * 60 + int(third)
            )
            timestamps.add(float(seconds))
    return sorted(timestamps)


def _label(timestamp: float) -> str:
    total = max(0, int(round(timestamp)))
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _deduplicate_timestamps(values: Iterable[float], *, gap_s: float = 0.5) -> list[float]:
    selected: list[float] = []
    for value in sorted(max(0.0, float(item)) for item in values):
        if not selected or value - selected[-1] >= gap_s:
            selected.append(value)
    return selected


def extract_keyframes(
    clip_path: str | Path,
    *,
    max_keyframes: int = 8,
    mode: str = "uniform",
    cited_timestamps: Iterable[float] | None = None,
    jpeg_quality: int = 85,
) -> KeyframeSet:
    """Extract benchmark frames from one walkthrough clip.

    ``uniform`` is the default. ``uniform+cited`` and ``cited`` are explicit
    question-aware ablations and are deliberately not enabled automatically.
    """
    path = Path(clip_path).resolve()
    if mode == "none":
        return KeyframeSet(mode=mode, clip_path=path)
    if mode not in {"uniform", "cited", "uniform+cited"}:
        raise ValueError(f"Unsupported keyframe mode: {mode}")
    if max_keyframes <= 0:
        return KeyframeSet(mode=mode, clip_path=path)
    if not path.is_file():
        raise FileNotFoundError(f"Video clip not found: {path}")
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Video extraction requires the video extra: pip install -e '.[video]'"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open clip: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"Clip has invalid FPS/frame count: {path}")
        duration = total_frames / fps
        uniform: list[float] = []
        if mode in {"uniform", "uniform+cited"}:
            uniform = [duration * (index + 0.5) / max_keyframes for index in range(max_keyframes)]
        cited = [value for value in (cited_timestamps or []) if 0 <= float(value) <= duration]
        if mode == "uniform":
            requested = uniform
        elif mode == "cited":
            requested = cited[:max_keyframes]
        else:
            anchors = _deduplicate_timestamps(cited)
            requested = anchors + uniform[: max(0, max_keyframes - len(anchors))]
        timestamps = _deduplicate_timestamps(requested)[:max_keyframes]

        images: list[bytes] = []
        labels: list[str] = []
        accepted_timestamps: list[float] = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            )
            if not encoded:
                continue
            images.append(buffer.tobytes())
            labels.append(_label(timestamp))
            accepted_timestamps.append(timestamp)
        return KeyframeSet(
            images=images,
            labels=labels,
            timestamps=accepted_timestamps,
            mode=mode,
            clip_path=path,
        )
    finally:
        capture.release()
