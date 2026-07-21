from __future__ import annotations

import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schemas import VisualClawRound

_BBOX_RE = re.compile(r"\\bbox\{\s*([A-Z])\s*\}", re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*Answer\s*[:=]\s*([A-Z])\b", re.IGNORECASE)
_THE_ANSWER_RE = re.compile(r"\bthe\s+answer\s+is\s+\(?\s*([A-Z])\b", re.IGNORECASE)
_SOLITARY_RE = re.compile(r"(?:^|[\s\(\[])([A-Z])(?:[\s\.\)\]\.,]|$)")


@dataclass(slots=True)
class MultiChoiceScore:
    passed: bool
    score: float
    extracted: str | None
    expected: list[str]
    parse_method: str
    raw_response_tail: str


@dataclass(slots=True)
class ExecCheckScore:
    passed: bool
    score: float
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    command_executed: str
    expect_exit: int
    parse_notes: list[str] = field(default_factory=list)
    skipped: bool = False


def extract_choice(response: str) -> tuple[str | None, str]:
    if not response:
        return None, "empty_response"
    for pattern, method in (
        (_BBOX_RE, "bbox"),
        (_ANSWER_LINE_RE, "answer_line"),
        (_THE_ANSWER_RE, "the_answer_is"),
    ):
        match = pattern.search(response)
        if match:
            return match.group(1).upper(), method
    matches = _SOLITARY_RE.findall(response[-300:])
    if matches:
        return matches[-1].upper(), "solitary_letter_tail"
    return None, "unparseable"


def _expected_choices(evaluation: dict[str, Any]) -> list[str]:
    expected = evaluation.get("answer")
    if expected is None:
        expected = evaluation.get("answers")
    if isinstance(expected, str):
        expected = [expected]
    if not isinstance(expected, list):
        return []
    normalized: list[str] = []
    for value in expected:
        text = str(value).strip().upper()
        if len(text) == 1 and text.isalpha():
            normalized.append(text)
    return normalized


def score_multi_choice(round_item: VisualClawRound, response: str) -> MultiChoiceScore:
    expected = _expected_choices(round_item.evaluation)
    extracted, method = extract_choice(response)
    passed = extracted is not None and extracted in expected
    return MultiChoiceScore(
        passed=passed,
        score=1.0 if passed else 0.0,
        extracted=extracted,
        expected=expected,
        parse_method=method,
        raw_response_tail=(response or "")[-200:],
    )


def _expand_placeholders(
    command: str,
    *,
    eval_dir: Path,
    agent_id: str,
    workspace: Path,
) -> str:
    return (
        command.replace("${eval_dir}", str(eval_dir.resolve()))
        .replace("${agent_id}", agent_id)
        .replace("${workspace}", str(workspace.resolve()))
    )


def _argv(command: str) -> list[str]:
    argv = shlex.split(command)
    if argv and Path(argv[0]).name in {"python", "python3"} and len(Path(argv[0]).parts) == 1:
        argv[0] = sys.executable
    return argv


def score_exec_check(
    round_item: VisualClawRound,
    *,
    eval_dir: Path,
    agent_id: str,
    workspace: Path,
    checker_mode: str = "host",
) -> ExecCheckScore:
    evaluation = round_item.evaluation
    raw_command = str(evaluation.get("command") or "")
    expect_exit = int(evaluation.get("expect_exit", 0))
    if checker_mode == "disabled":
        return ExecCheckScore(
            passed=False,
            score=0.0,
            exit_code=None,
            stdout_tail="",
            stderr_tail="",
            command_executed="",
            expect_exit=expect_exit,
            parse_notes=["checker execution disabled"],
            skipped=True,
        )
    if checker_mode != "host":
        raise ValueError(f"Unsupported checker mode: {checker_mode}")
    if not raw_command:
        return ExecCheckScore(
            passed=False,
            score=0.0,
            exit_code=None,
            stdout_tail="",
            stderr_tail="",
            command_executed="",
            expect_exit=expect_exit,
            parse_notes=["eval.command is empty"],
        )

    expanded = _expand_placeholders(
        raw_command,
        eval_dir=eval_dir,
        agent_id=agent_id,
        workspace=workspace,
    )
    argv = _argv(expanded)
    command_executed = shlex.join(argv) if argv else expanded
    timeout = float(evaluation.get("timeout", evaluation.get("timeout_s", 30)))
    try:
        process = subprocess.run(
            argv,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExecCheckScore(
            passed=False,
            score=0.0,
            exit_code=None,
            stdout_tail=stdout[-1000:],
            stderr_tail=stderr[-1000:],
            command_executed=command_executed,
            expect_exit=expect_exit,
            parse_notes=[f"timeout after {timeout}s"],
        )
    except (FileNotFoundError, OSError) as exc:
        return ExecCheckScore(
            passed=False,
            score=0.0,
            exit_code=None,
            stdout_tail="",
            stderr_tail=str(exc),
            command_executed=command_executed,
            expect_exit=expect_exit,
            parse_notes=[f"checker launch failed: {exc}"],
        )

    notes: list[str] = []
    passed = process.returncode == expect_exit
    expected_stdout = evaluation.get("expect_stdout")
    if expected_stdout is not None and str(expected_stdout) not in process.stdout:
        passed = False
        notes.append("expect_stdout substring missing")
    expected_regex = evaluation.get("expect_stdout_regex")
    if expected_regex is not None and not re.search(str(expected_regex), process.stdout):
        passed = False
        notes.append("expect_stdout_regex did not match")
    return ExecCheckScore(
        passed=passed,
        score=1.0 if passed else 0.0,
        exit_code=process.returncode,
        stdout_tail=process.stdout[-1000:],
        stderr_tail=process.stderr[-1000:],
        command_executed=command_executed,
        expect_exit=expect_exit,
        parse_notes=notes,
    )


def score_round(
    round_item: VisualClawRound,
    response: str,
    *,
    eval_dir: Path,
    agent_id: str,
    workspace: Path,
    checker_mode: str = "host",
) -> dict[str, Any]:
    if round_item.round_type == "multi_choice":
        score = score_multi_choice(round_item, response)
    else:
        score = score_exec_check(
            round_item,
            eval_dir=eval_dir,
            agent_id=agent_id,
            workspace=workspace,
            checker_mode=checker_mode,
        )
    return asdict(score)
