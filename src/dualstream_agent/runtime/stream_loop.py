from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from PIL import Image

from dualstream_agent.runtime.session import DualStreamSession
from dualstream_agent.schemas import RuntimeOutput


async def image_directory_source(
    directory: str | Path,
    *,
    fps: float = 1.0,
    realtime: bool = False,
) -> AsyncIterator[tuple[float, Image.Image]]:
    paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    interval = 1.0 / max(fps, 1e-6)
    for index, path in enumerate(paths):
        if realtime and index:
            await asyncio.sleep(interval)
        yield index * interval, Image.open(path).convert("RGB")


async def opencv_video_source(
    source: str | int,
    *,
    sample_fps: float = 1.0,
    realtime: bool = False,
) -> AsyncIterator[tuple[float, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install video support with: pip install -e '.[video]'") from exc

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
    stride = max(1, round(native_fps / max(sample_fps, 1e-6)))
    frame_index = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                timestamp = frame_index / native_fps
                if realtime:
                    target = started + timestamp
                    await asyncio.sleep(max(0.0, target - time.monotonic()))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield timestamp, rgb
            frame_index += 1
    finally:
        capture.release()


async def run_stream(
    session: DualStreamSession,
    source: AsyncIterator[tuple[float, Any]],
    *,
    user_goal: str = "",
    on_output: Callable[[RuntimeOutput], Any] | None = None,
) -> list[RuntimeOutput]:
    outputs: list[RuntimeOutput] = []
    async for timestamp, image in source:
        output = await session.process_frame(
            image,
            timestamp=timestamp,
            user_goal=user_goal,
        )
        outputs.append(output)
        if on_output is not None:
            maybe_awaitable = on_output(output)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
    return outputs
