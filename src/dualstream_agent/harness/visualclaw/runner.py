from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dualstream_agent.config import AppConfig

from .agent import VisualClawBaselineAgent
from .loader import VisualClawDataset
from .prompt import apply_write_blocks, build_round_prompt
from .schemas import KeyframeSet, VisualClawRound, VisualClawScenario
from .scoring import score_round
from .video import extract_keyframes, parse_cited_timestamps
from .workspace import apply_updates, stage_keyframes, stage_work_copy

_CLIP_REFERENCE_RE = re.compile(r"\[clip\s*@\s*\d{1,2}:\d{2}(?::\d{2})?\]", re.IGNORECASE)


@dataclass(slots=True)
class VisualClawRunOptions:
    output_root: Path = Path("runs/visualclaw")
    run_id: str | None = None
    release_only: bool = True
    include_deprecated: bool = False
    video_required_only: bool = False
    round_types: tuple[str, ...] = ()
    round_ids: tuple[str, ...] = ()
    max_rounds: int = 0
    max_scenarios: int = 0
    max_keyframes: int = 8
    keyframe_mode: str = "uniform"
    allow_missing_clips: bool = False
    checker_mode: str = "host"
    dry_run: bool = False
    resume: bool = False
    fail_fast: bool = False
    workspace_char_budget: int = 20_000
    sessions_char_budget: int = 10_000
    max_new_tokens: int = 1024
    temperature: float = 0.0
    agent_id: str = "."


@dataclass(slots=True)
class _Aggregate:
    scenarios: int = 0
    rounds: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    multi_choice_rounds: int = 0
    exec_check_rounds: int = 0
    total_latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def observe(self, scenario_result: dict[str, Any]) -> None:
        self.scenarios += 1
        for item in scenario_result.get("results", []):
            self.rounds += 1
            if item.get("type") == "multi_choice":
                self.multi_choice_rounds += 1
            elif item.get("type") == "exec_check":
                self.exec_check_rounds += 1
            if item.get("skipped"):
                self.skipped += 1
            elif item.get("passed"):
                self.passed += 1
            else:
                self.failed += 1
            self.total_latency_s += float(item.get("latency_sec") or 0.0)
            if item.get("error"):
                self.errors.append(f"{scenario_result.get('scenario_id')}:{item.get('id')}: {item['error']}")

    def summary(self) -> dict[str, Any]:
        scored = self.passed + self.failed
        return {
            "scenarios": self.scenarios,
            "rounds": self.rounds,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "accuracy": self.passed / scored if scored else 0.0,
            "multi_choice_rounds": self.multi_choice_rounds,
            "exec_check_rounds": self.exec_check_rounds,
            "total_latency_s": self.total_latency_s,
            "mean_latency_s": self.total_latency_s / self.rounds if self.rounds else 0.0,
            "errors": self.errors,
        }


class VisualClawArenaRunner:
    """Full VisualClawArena release adapter for local DualStreamAgent backends."""

    def __init__(
        self,
        config: AppConfig,
        dataset_root: str | Path,
        options: VisualClawRunOptions | None = None,
        *,
        agent: VisualClawBaselineAgent | None = None,
    ):
        self.config = config
        self.dataset = VisualClawDataset(dataset_root)
        self.options = options or VisualClawRunOptions()
        self.options.output_root = Path(self.options.output_root).expanduser().resolve()
        self.options.output_root.mkdir(parents=True, exist_ok=True)
        self.agent = agent or VisualClawBaselineAgent(
            config,
            max_new_tokens=self.options.max_new_tokens,
            temperature=self.options.temperature,
        )
        self._owns_agent = agent is None
        self.batch_run_id = self.options.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _load(self, scenario_id: str) -> VisualClawScenario:
        return self.dataset.load_scenario(
            scenario_id,
            release_only=self.options.release_only,
            include_deprecated=self.options.include_deprecated,
            video_required_only=self.options.video_required_only,
            round_types=self.options.round_types,
            round_ids=self.options.round_ids,
            max_rounds=self.options.max_rounds,
        )

    @staticmethod
    def _rounds_reference_clip(rounds: Iterable[VisualClawRound]) -> bool:
        return any(_CLIP_REFERENCE_RE.search(f"{item.question} {item.feedback}") for item in rounds)

    def _keyframes(self, scenario: VisualClawScenario) -> KeyframeSet:
        if self.options.keyframe_mode == "none":
            return KeyframeSet(mode="none")
        if not scenario.clip_paths:
            if self._rounds_reference_clip(scenario.rounds) and not self.options.allow_missing_clips:
                raise FileNotFoundError(
                    f"{scenario.scenario_id} references clip timestamps but no video was found"
                )
            return KeyframeSet(mode="none")
        cited = parse_cited_timestamps(scenario.rounds)
        return extract_keyframes(
            scenario.clip_paths[0],
            max_keyframes=self.options.max_keyframes,
            mode=self.options.keyframe_mode,
            cited_timestamps=cited,
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _append_jsonl(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    async def run_scenario(self, scenario_id: str) -> dict[str, Any]:
        scenario = self._load(scenario_id)
        run_dir = self.options.output_root / scenario_id / self.batch_run_id
        result_path = run_dir / "results.json"
        if self.options.resume and result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))

        work_copy = stage_work_copy(scenario, run_dir)
        keyframes = self._keyframes(scenario)
        staged_frames = stage_keyframes(work_copy, keyframes)
        results: list[dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()

        for round_item in scenario.rounds:
            round_started = time.perf_counter()
            log_dir = work_copy.agent_logs / round_item.round_id
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                applied = apply_updates(work_copy, round_item.update_ids)
                prompt = build_round_prompt(
                    round_item,
                    workspace=work_copy.workspace,
                    sessions=work_copy.sessions,
                    keyframe_labels=keyframes.labels,
                    workspace_char_budget=self.options.workspace_char_budget,
                    sessions_char_budget=self.options.sessions_char_budget,
                )
                (log_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                turn = await self.agent.run_round(
                    round_item,
                    prompt=prompt,
                    images=keyframes.images,
                    dry_run=self.options.dry_run,
                )
                (log_dir / "answer.txt").write_text(turn.response or "", encoding="utf-8")
                files_written = (
                    apply_write_blocks(turn.response, work_copy.workspace)
                    if round_item.round_type == "exec_check"
                    else []
                )
                score = score_round(
                    round_item,
                    turn.response,
                    eval_dir=scenario.spec_dir,
                    agent_id=self.options.agent_id,
                    workspace=work_copy.workspace,
                    checker_mode=self.options.checker_mode,
                )
                latency = turn.latency_s
                if latency is None:
                    latency = time.perf_counter() - round_started
                result = {
                    "id": round_item.round_id,
                    "round_id": round_item.round_id,
                    "round_number": round_item.round_number,
                    "type": round_item.round_type,
                    "passed": bool(score.get("passed")),
                    "score": float(score.get("score", 0.0)),
                    "skipped": bool(score.get("skipped", False)),
                    "response": turn.response,
                    "response_chars": len(turn.response or ""),
                    "latency_sec": float(latency),
                    "usage": {
                        "input_tokens": turn.prompt_tokens,
                        "output_tokens": turn.completion_tokens,
                    },
                    "files_written_by_agent": files_written,
                    "applied_updates": [asdict(item) for item in applied],
                    "required_modalities": round_item.required_modalities,
                    "required_skills": round_item.required_skills,
                    "video_required": round_item.video_required,
                    "evidence_type": round_item.evidence_type,
                    **score,
                }
            except Exception as exc:
                result = {
                    "id": round_item.round_id,
                    "round_id": round_item.round_id,
                    "round_number": round_item.round_number,
                    "type": round_item.round_type,
                    "passed": False,
                    "score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_sec": time.perf_counter() - round_started,
                }
                (log_dir / "error.txt").write_text(result["error"], encoding="utf-8")
                if self.options.fail_fast:
                    results.append(result)
                    raise
            results.append(result)
            self._append_jsonl(work_copy.access_logs / "rounds.jsonl", result)
            self._write_json(log_dir / "result.json", result)

        passed = sum(1 for item in results if item.get("passed"))
        skipped = sum(1 for item in results if item.get("skipped"))
        scored = max(0, len(results) - skipped)
        scenario_result = {
            "benchmark": "VisualClawArena",
            "dataset_version": self.dataset.version,
            "scenario_id": scenario.scenario_id,
            "source_bucket": scenario.source_bucket,
            "run_id": self.batch_run_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(work_copy.workspace),
            "clip_paths": [str(path) for path in scenario.clip_paths],
            "keyframe_mode": keyframes.mode,
            "keyframe_labels": keyframes.labels,
            "staged_keyframes": [str(path.relative_to(work_copy.workspace)) for path in staged_frames],
            "rounds": len(results),
            "passed": passed,
            "skipped": skipped,
            "accuracy": passed / scored if scored else 0.0,
            "results": results,
        }
        self._write_json(result_path, scenario_result)
        return scenario_result

    async def run(self, scenario_ids: Iterable[str] | None = None) -> dict[str, Any]:
        selected = list(scenario_ids or self.dataset.list_scenario_ids())
        if self.options.max_scenarios > 0:
            selected = selected[: self.options.max_scenarios]
        aggregate = _Aggregate()
        scenario_results: list[dict[str, Any]] = []
        try:
            for scenario_id in selected:
                try:
                    result = await self.run_scenario(scenario_id)
                except Exception as exc:
                    result = {
                        "benchmark": "VisualClawArena",
                        "scenario_id": scenario_id,
                        "run_id": self.batch_run_id,
                        "results": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    aggregate.errors.append(f"{scenario_id}: {result['error']}")
                    if self.options.fail_fast:
                        raise
                scenario_results.append(result)
                aggregate.observe(result)
        finally:
            if self._owns_agent:
                await self.agent.close()

        summary = {
            "benchmark": "VisualClawArena",
            "dataset_root": str(self.dataset.root),
            "dataset_version": self.dataset.version,
            "run_id": self.batch_run_id,
            "options": asdict(self.options),
            "metrics": aggregate.summary(),
            "scenario_results": [
                {
                    "scenario_id": item.get("scenario_id"),
                    "rounds": item.get("rounds", 0),
                    "passed": item.get("passed", 0),
                    "accuracy": item.get("accuracy", 0.0),
                    "error": item.get("error"),
                }
                for item in scenario_results
            ],
        }
        summary_path = self.options.output_root / f"summary-{self.batch_run_id}.json"
        self._write_json(summary_path, summary)
        per_round_path = self.options.output_root / f"per-round-{self.batch_run_id}.jsonl"
        if per_round_path.exists():
            per_round_path.unlink()
        for scenario_result in scenario_results:
            for result in scenario_result.get("results", []):
                self._append_jsonl(
                    per_round_path,
                    {"scenario_id": scenario_result.get("scenario_id"), **result},
                )
        summary["summary_path"] = str(summary_path)
        summary["per_round_path"] = str(per_round_path)
        return summary
