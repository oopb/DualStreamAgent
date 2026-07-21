from __future__ import annotations

import json
from pathlib import Path

import pytest

from dualstream_agent.config import AppConfig
from dualstream_agent.harness.visualclaw.loader import VisualClawDataset
from dualstream_agent.harness.visualclaw.runner import VisualClawArenaRunner, VisualClawRunOptions
from dualstream_agent.harness.visualclaw.schemas import AgentTurn, VisualClawRound
from dualstream_agent.harness.visualclaw.scoring import extract_choice, score_exec_check, score_multi_choice
from dualstream_agent.harness.visualclaw.workspace import apply_updates, stage_work_copy


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_release(root: Path) -> Path:
    _write_json(root / "manifest.json", {"dataset": "VisualClawArena", "version": "test-v1"})
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "scenarios.jsonl").write_text(
        json.dumps(
            {
                "scenario_id": "demo",
                "scenario_data_path": "scenarios/demo/data",
                "scenario_spec_path": "scenarios/demo/spec",
                "source_bucket": "synthetic",
                "clip_paths": ["scenarios/demo/data/clip/clip.mp4"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data = root / "scenarios" / "demo" / "data"
    spec = root / "scenarios" / "demo" / "spec"
    (data / "workspace").mkdir(parents=True)
    (data / "workspace" / "input.txt").write_text("initial", encoding="utf-8")
    (data / "workspace" / "delete_me.txt").write_text("remove", encoding="utf-8")
    (data / "sessions").mkdir(parents=True)
    (data / "sessions" / "history.jsonl").write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")
    (data / "clip").mkdir(parents=True)
    (data / "clip" / "clip.mp4").write_bytes(b"synthetic-placeholder")

    update = data / "updates" / "u1"
    update.mkdir(parents=True)
    (update / "replacement.txt").write_text("updated", encoding="utf-8")
    (update / "session_line.txt").write_text("next-session-line\n", encoding="utf-8")
    _write_json(
        update / "update_manifest.json",
        {
            "files": [
                {"action": "replace", "source": "replacement.txt", "target": "input.txt"},
                {"action": "delete", "target": "delete_me.txt"},
            ],
            "session_appends": [
                {
                    "action": "append",
                    "source": "session_line.txt",
                    "target": "history.jsonl",
                }
            ],
        },
    )

    checker = spec / "scripts" / "check_output.py"
    checker.parent.mkdir(parents=True)
    checker.write_text(
        """from pathlib import Path
import sys
workspace = Path(sys.argv[1])
ok = (workspace / 'answer.txt').read_text(encoding='utf-8') == 'done\n'
print('PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
""",
        encoding="utf-8",
    )
    _write_json(
        spec / "questions.json",
        {
            "rounds": [
                {
                    "id": "mc1",
                    "round_number": 1,
                    "type": "multi_choice",
                    "question": "Choose the correct option.",
                    "eval": {"options": {"A": "right", "B": "wrong"}, "answer": ["A"]},
                    "included_in_release_eval": True,
                },
                {
                    "id": "ec1",
                    "round_number": 2,
                    "type": "exec_check",
                    "question": "Write answer.txt containing done.",
                    "update_ids": ["u1"],
                    "eval": {
                        "command": "python ${eval_dir}/${agent_id}/scripts/check_output.py ${workspace}",
                        "expect_exit": 0,
                        "expect_stdout": "PASS",
                    },
                    "included_in_release_eval": True,
                },
                {
                    "id": "old",
                    "round_number": 3,
                    "type": "multi_choice",
                    "question": "Deprecated round.",
                    "eval": {"answer": ["B"]},
                    "deprecated": True,
                },
                {
                    "id": "private",
                    "round_number": 4,
                    "type": "multi_choice",
                    "question": "Not in release evaluation.",
                    "eval": {"answer": ["C"]},
                    "included_in_release_eval": False,
                },
            ]
        },
    )
    return root


def test_loader_filters_release_rounds(tmp_path: Path) -> None:
    dataset = VisualClawDataset(_build_release(tmp_path / "release"))
    scenario = dataset.load_scenario("demo")
    assert dataset.version == "test-v1"
    assert [item.round_id for item in scenario.rounds] == ["mc1", "ec1"]
    all_rounds = dataset.load_scenario(
        "demo", release_only=False, include_deprecated=True
    )
    assert [item.round_id for item in all_rounds.rounds] == ["mc1", "ec1", "old", "private"]
    report = dataset.validate()
    assert report["valid"] is True
    assert report["rounds_loaded"] == 4


def test_workspace_updates_are_stateful_and_scoped(tmp_path: Path) -> None:
    dataset = VisualClawDataset(_build_release(tmp_path / "release"))
    scenario = dataset.load_scenario("demo")
    work_copy = stage_work_copy(scenario, tmp_path / "run")
    records = apply_updates(work_copy, ["u1"])
    assert {item.action for item in records} == {"replace", "delete", "append"}
    assert (work_copy.workspace / "input.txt").read_text(encoding="utf-8") == "updated"
    assert not (work_copy.workspace / "delete_me.txt").exists()
    assert "next-session-line" in (work_copy.sessions / "history.jsonl").read_text(encoding="utf-8")


def test_official_style_scorers(tmp_path: Path) -> None:
    choice_round = VisualClawRound.from_dict(
        {"id": "q", "type": "multi_choice", "question": "q", "eval": {"answer": ["C"]}}
    )
    assert extract_choice("reasoning\n\\bbox{C}") == ("C", "bbox")
    assert score_multi_choice(choice_round, "reasoning\n\\bbox{C}").passed

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("done\n", encoding="utf-8")
    eval_dir = tmp_path / "spec"
    script = eval_dir / "scripts" / "check.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from pathlib import Path\nimport sys\nprint('PASS')\nraise SystemExit(0 if Path(sys.argv[1]).exists() else 1)\n",
        encoding="utf-8",
    )
    exec_round = VisualClawRound.from_dict(
        {
            "id": "e",
            "type": "exec_check",
            "question": "write",
            "eval": {
                "command": "python ${eval_dir}/${agent_id}/scripts/check.py ${workspace}/answer.txt",
                "expect_stdout_regex": "PASS",
            },
        }
    )
    score = score_exec_check(
        exec_round,
        eval_dir=eval_dir,
        agent_id=".",
        workspace=workspace,
    )
    assert score.passed


class _ScriptedAgent:
    async def run_round(self, round_item, *, prompt, images, dry_run=False):
        assert "Current persistent workspace" in prompt
        if round_item.round_type == "multi_choice":
            return AgentTurn(response="Reasoning.\n\\bbox{A}")
        assert "updated" in prompt
        assert "delete_me.txt" not in prompt
        return AgentTurn(
            response="### WRITE_FILE: answer.txt\n```text\ndone\n```"
        )

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_runner_executes_stateful_rounds_and_checkers(tmp_path: Path) -> None:
    root = _build_release(tmp_path / "release")
    options = VisualClawRunOptions(
        output_root=tmp_path / "runs",
        run_id="test-run",
        keyframe_mode="none",
        checker_mode="host",
    )
    runner = VisualClawArenaRunner(
        AppConfig(), root, options, agent=_ScriptedAgent()
    )
    summary = await runner.run(["demo"])
    assert summary["metrics"]["rounds"] == 2
    assert summary["metrics"]["passed"] == 2
    result_path = tmp_path / "runs" / "demo" / "test-run" / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["accuracy"] == 1.0
    assert result["results"][1]["files_written_by_agent"] == ["answer.txt"]
    assert result["results"][1]["applied_updates"]
