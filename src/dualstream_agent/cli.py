from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from dualstream_agent.config import load_config
from dualstream_agent.harness.runner import ScenarioRunner
from dualstream_agent.runtime.session import DualStreamSession
from dualstream_agent.runtime.stream_loop import (
    image_directory_source,
    opencv_video_source,
    run_stream,
)


def _print_output(output) -> None:
    payload = {
        "timestamp": output.timestamp,
        "response": output.response,
        "decision": asdict(output.decision),
        "signal": asdict(output.signal),
        "trace_id": output.trace_id,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str))


async def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.command == "scenario":
        result = await ScenarioRunner(config, args.output).run(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    session = DualStreamSession(config)
    try:
        if args.command == "text":
            _print_output(await session.process_text(args.text))
        elif args.command == "images":
            await run_stream(
                session,
                image_directory_source(args.path, fps=args.fps, realtime=args.realtime),
                user_goal=args.goal,
                on_output=_print_output,
            )
        elif args.command == "video":
            source: str | int = int(args.path) if args.path.isdigit() else args.path
            await run_stream(
                session,
                opencv_video_source(source, sample_fps=args.fps, realtime=args.realtime),
                user_goal=args.goal,
                on_output=_print_output,
            )
    finally:
        await session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dualstream")
    parser.add_argument("--config", default="configs/mock.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text = subparsers.add_parser("text", help="Run one text request")
    text.add_argument("text")

    images = subparsers.add_parser("images", help="Replay an image directory")
    images.add_argument("path")
    images.add_argument("--goal", default="")
    images.add_argument("--fps", type=float, default=1.0)
    images.add_argument("--realtime", action="store_true")

    video = subparsers.add_parser("video", help="Run a video file or camera index")
    video.add_argument("path")
    video.add_argument("--goal", default="")
    video.add_argument("--fps", type=float, default=1.0)
    video.add_argument("--realtime", action="store_true")

    scenario = subparsers.add_parser("scenario", help="Run a harness scenario")
    scenario.add_argument("path")
    scenario.add_argument("--output", default="runs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
