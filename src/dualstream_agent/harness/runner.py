from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from dualstream_agent.config import AppConfig
from dualstream_agent.harness.checker import check_expectations
from dualstream_agent.harness.metrics import HarnessMetrics
from dualstream_agent.harness.scenario import load_scenario
from dualstream_agent.runtime.session import DualStreamSession


class ScenarioRunner:
    def __init__(self, config: AppConfig, output_root: str | Path = "runs"):
        self.config = config
        self.output_root = Path(output_root)

    async def run(self, scenario_path: str | Path) -> dict[str, Any]:
        scenario = load_scenario(scenario_path)
        run_id = f"{scenario.name}-{int(time.time())}"
        workspace = self.output_root / run_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        initial = scenario.root / "workspace"
        if initial.exists():
            shutil.copytree(initial, workspace, dirs_exist_ok=True)

        from dualstream_agent.tools.builtins import build_default_tools

        session = DualStreamSession(
            self.config,
            session_id=run_id,
            tools=build_default_tools(workspace),
        )
        metrics = HarnessMetrics()
        rendered_outputs: list[dict[str, Any]] = []
        try:
            for event in scenario.events:
                started = event.at
                if event.kind == "frame":
                    image_path = scenario.root / str(event.payload["path"])
                    output = await session.process_frame(
                        Image.open(image_path).convert("RGB"),
                        timestamp=event.at,
                        user_goal=event.payload.get("user_goal"),
                        source="scenario",
                        metadata={"path": str(image_path)},
                    )
                elif event.kind == "message":
                    output = await session.process_text(
                        str(event.payload.get("text", "")), timestamp=event.at
                    )
                elif event.kind == "copy":
                    source = scenario.root / str(event.payload["source"])
                    target = workspace / str(event.payload["target"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    continue
                else:
                    raise ValueError(f"Unsupported scenario event kind: {event.kind}")

                metrics.observe(
                    output,
                    event_started_at=started,
                    expect_response=event.payload.get("expect_response"),
                )
                rendered_outputs.append(
                    {
                        "timestamp": output.timestamp,
                        "response": output.response,
                        "decision": asdict(output.decision),
                        "signal": asdict(output.signal),
                        "trace_id": output.trace_id,
                    }
                )
        finally:
            await session.close()

        checks = check_expectations(
            rendered_outputs,
            scenario.expected,
            workspace=workspace,
        )
        result = {
            "scenario": scenario.name,
            "run_id": run_id,
            "workspace": str(workspace),
            "metrics": metrics.summary(),
            "checks": checks,
            "outputs": rendered_outputs,
        }
        result_path = self.output_root / run_id / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
