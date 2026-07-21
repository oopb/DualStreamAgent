from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dualstream_agent.config import load_config

from .loader import VisualClawDataset, download_visualclaw_arena
from .runner import VisualClawArenaRunner, VisualClawRunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dualstream-visualclaw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download VisualClawArena from Hugging Face")
    download.add_argument("output")
    download.add_argument("--revision", default="main")
    download.add_argument("--token", default=None)

    validate = subparsers.add_parser("validate", help="Validate an unpacked VisualClawArena release")
    validate.add_argument("dataset_root")
    validate.add_argument("--max-scenarios", type=int, default=0)

    run = subparsers.add_parser("run", help="Run VisualClawArena scenarios")
    run.add_argument("dataset_root")
    run.add_argument("--config", default="configs/mock.yaml")
    run.add_argument("--scenario", action="append", default=[])
    run.add_argument("--all", action="store_true", help="Run all scenarios when --scenario is omitted")
    run.add_argument("--output", default="runs/visualclaw")
    run.add_argument("--run-id", default=None)
    run.add_argument("--max-scenarios", type=int, default=0)
    run.add_argument("--max-rounds", type=int, default=0)
    run.add_argument("--round-id", action="append", default=[])
    run.add_argument("--round-type", action="append", choices=["multi_choice", "exec_check"], default=[])
    run.add_argument("--release-only", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--include-deprecated", action="store_true")
    run.add_argument("--video-required-only", action="store_true")
    run.add_argument("--keyframe-mode", choices=["uniform", "cited", "uniform+cited", "none"], default="uniform")
    run.add_argument("--max-keyframes", type=int, default=8)
    run.add_argument("--allow-missing-clips", action="store_true")
    run.add_argument("--checker-mode", choices=["host", "disabled"], default="host")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--workspace-char-budget", type=int, default=20_000)
    run.add_argument("--sessions-char-budget", type=int, default=10_000)
    run.add_argument("--max-new-tokens", type=int, default=1024)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--agent-id", default=".")
    return parser


async def _run(args: argparse.Namespace) -> None:
    if args.command == "download":
        root = download_visualclaw_arena(args.output, revision=args.revision, token=args.token)
        print(json.dumps({"dataset_root": str(root)}, indent=2))
        return
    if args.command == "validate":
        result = VisualClawDataset(args.dataset_root).validate(max_scenarios=args.max_scenarios)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not args.scenario and not args.all:
        raise SystemExit("Pass one or more --scenario values, or use --all")
    config = load_config(args.config)
    options = VisualClawRunOptions(
        output_root=Path(args.output),
        run_id=args.run_id,
        release_only=args.release_only,
        include_deprecated=args.include_deprecated,
        video_required_only=args.video_required_only,
        round_types=tuple(args.round_type),
        round_ids=tuple(args.round_id),
        max_rounds=args.max_rounds,
        max_scenarios=args.max_scenarios,
        max_keyframes=args.max_keyframes,
        keyframe_mode=args.keyframe_mode,
        allow_missing_clips=args.allow_missing_clips,
        checker_mode=args.checker_mode,
        dry_run=args.dry_run,
        resume=args.resume,
        fail_fast=args.fail_fast,
        workspace_char_budget=args.workspace_char_budget,
        sessions_char_budget=args.sessions_char_budget,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        agent_id=args.agent_id,
    )
    runner = VisualClawArenaRunner(config, args.dataset_root, options)
    result = await runner.run(args.scenario or None)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
