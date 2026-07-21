from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def check_expectations(
    outputs: list[dict[str, Any]],
    expected: dict[str, Any],
    *,
    workspace: str | Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    minimum_responses = int(expected.get("minimum_responses", 0))
    response_count = sum(1 for item in outputs if item.get("response"))
    checks["minimum_responses"] = {
        "passed": response_count >= minimum_responses,
        "actual": response_count,
        "expected": minimum_responses,
    }

    required_substrings = [str(value) for value in expected.get("response_contains", [])]
    joined = "\n".join(str(item.get("response", "")) for item in outputs).lower()
    checks["response_contains"] = {
        "passed": all(value.lower() in joined for value in required_substrings),
        "missing": [value for value in required_substrings if value.lower() not in joined],
    }

    command = expected.get("command")
    if command:
        completed = subprocess.run(
            command,
            cwd=Path(workspace),
            shell=True,
            capture_output=True,
            text=True,
            timeout=float(expected.get("command_timeout_s", 30)),
        )
        checks["command"] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    checks["passed"] = all(
        value.get("passed", True) for key, value in checks.items() if key != "passed"
    )
    return checks
