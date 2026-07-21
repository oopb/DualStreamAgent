# DualStreamAgent

DualStreamAgent is a local-first framework for an **evolvable, causal, real-time streaming, dual-system multimodal agent**. It separates fast perception and response timing from slower reasoning, memory, skills, and tool use.

The current release is a complete runnable research MVP. It is intentionally smaller than VisualClaw: one runtime path is shared by live execution and the benchmark harness, and the core package has no dependency on Claude Code, Codex, OpenClaw, commercial model providers, calendars, glasses, or online RL.

## Architecture

```text
video / images / user events
          |
          v
cheap causal change detector ----> skip unchanged frames
          |
          v
S1 fast system
  - current observation
  - relevance / novelty / urgency
  - speak now?
  - invoke S2?
          |
          v
DualController
  - WAIT
  - RESPOND
  - INTERRUPT
  - INVOKE_S2 / MEMORY / TOOL
          |
          v
S2 slow system
  - episodic memory retrieval
  - SKILL.md retrieval
  - bounded tool loop
          |
          v
response + JSONL trace + offline skill evolution
```

The response decision and the reasoning decision are separate. A frame can be important enough to invoke S2 while the system remains silent, and an urgent event can trigger an immediate S1 response without waiting for S2.

## Implemented modules

- Causal image/video stream runtime with frame buffering.
- Lightweight visual change gate with a maximum-silence ceiling.
- Structured S1 output and robust JSON extraction.
- Cooldown- and hysteresis-aware dual-system controller.
- S2 memory, skill, and tool orchestration.
- Persistent SQLite episodic memory.
- Human-readable `SKILL.md` bank with candidate and active states.
- Offline skill proposal and replay-based promotion interfaces.
- OpenAI-compatible and in-process Hugging Face Transformers backends.
- Image-directory, video-file, and camera stream sources.
- Scenario harness, machine checks, and response-timing metrics.
- JSONL traces and unit tests.

## Installation

Core runtime and mock backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

In-process Hugging Face Transformers backend:

```bash
pip install -e '.[transformers,video]'
```

OpenCV is optional and only required for video files or cameras.

## Backend choices

### 1. Transformers in the same Python process

Edit `configs/transformers.yaml`:

```yaml
s1:
  kind: transformers
  model: /path/to/local-vlm
  multimodal: true
  device_map: auto
  dtype: auto

s2:
  kind: transformers
  model: /path/to/local-vlm
  multimodal: true
  device_map: auto
  dtype: auto
```

When S1 and S2 configurations are identical, the session shares one loaded model instead of consuming memory for two copies. S1 and S2 still use different prompts and generation budgets.

The backend supports:

- text models through `AutoModelForCausalLM`;
- multimodal models through `AutoProcessor` and `AutoModelForImageTextToText`;
- an `AutoModelForVision2Seq` fallback for older Transformers versions;
- chat templates, local model paths, `device_map`, `dtype`, and `trust_remote_code`;
- asynchronous runtime calls through a worker thread so model generation does not block the event loop itself.

Model-specific processors still vary. A model should provide a working chat template and normal Transformers `generate()` behavior.

### 2. Local OpenAI-compatible server

Use `configs/openai-compatible.yaml` for vLLM, SGLang, llama.cpp server, or another compatible local endpoint:

```yaml
s1:
  kind: openai_compatible
  endpoint: http://127.0.0.1:8000/v1
  model: local-fast-vlm

s2:
  kind: openai_compatible
  endpoint: http://127.0.0.1:8001/v1
  model: local-reasoning-vlm
```

This mode is usually better for multi-process serving, continuous batching, and independent S1/S2 deployment.

## Quick start

Mock smoke test:

```bash
dualstream --config configs/mock.yaml text "Observe quietly unless something important happens."
```

Replay an image directory:

```bash
dualstream --config configs/transformers.yaml images ./frames \
  --goal "Tell me when the kettle starts boiling" \
  --fps 1
```

Run a video file causally:

```bash
dualstream --config configs/transformers.yaml video ./demo.mp4 \
  --goal "Warn me if the cup is close to falling" \
  --fps 2 \
  --realtime
```

Use a camera by passing its numeric index:

```bash
dualstream --config configs/transformers.yaml video 0 \
  --goal "Assist me while I assemble this object" \
  --fps 1 \
  --realtime
```

Run the included harness scenario:

```bash
dualstream --config configs/mock.yaml scenario examples/scenarios/basic
```

## Python API

```python
import asyncio
from PIL import Image

from dualstream_agent import DualStreamSession, load_config


async def main():
    config = load_config("configs/transformers.yaml")
    session = DualStreamSession(config)
    try:
        output = await session.process_frame(
            Image.open("frame.jpg"),
            timestamp=0.0,
            user_goal="Tell me only when something important changes.",
        )
        if output.response:
            print(output.response)
    finally:
        await session.close()


asyncio.run(main())
```

## S1 output protocol

S1 is instructed to return a structured object:

```json
{
  "summary": "The cup moved close to the table edge.",
  "event": "cup_moved",
  "relevance": 0.91,
  "confidence": 0.87,
  "urgency": 0.66,
  "novelty": 0.83,
  "need_reasoning": false,
  "need_memory": false,
  "need_tool": false,
  "should_respond": true,
  "response": "The cup is close to the edge.",
  "memory_note": "Cup moved close to the table edge."
}
```

The controller makes the final decision. The model does not bypass cooldown, urgency, or routing rules merely by emitting text.

## Skills and evolution

A skill is stored as `skills/<name>/SKILL.md` with YAML frontmatter:

```markdown
---
name: verify_temporal_order
description: Verify the order of events in a stream.
version: 1
status: active
tags: [temporal, video]
---

## Procedure

1. Locate the earliest reliable evidence.
2. Separate event start from event completion.
3. Verify the answer against timestamps.
```

The evolution path is deliberately offline:

```text
failed traces
    -> SkillEvolver.propose()
    -> status=candidate
    -> ReplayValidator.validate()
    -> promote only when utility improves
```

Generated skills are never activated directly in the live stream.

## Harness scenario format

```yaml
name: kettle-watch
events:
  - at: 0.0
    kind: frame
    payload:
      path: frames/000.jpg
      user_goal: Tell me when the kettle boils.
      expect_response: false
  - at: 4.0
    kind: frame
    payload:
      path: frames/004.jpg
      expect_response: true
expected:
  minimum_responses: 1
  response_contains: [boiling]
```

Supported event types are `frame`, `message`, and `copy`. Each scenario gets a persistent workspace and a `result.json` containing outputs, decisions, S1 signals, timing metrics, and checker results.

## Research limitations of the MVP

- The cheap change detector measures visual novelty, not deep task semantics; S1 supplies semantic judgment after the gate.
- The runtime currently processes accepted observations sequentially. Overlapping S1/S2 jobs, cancellation, backpressure, and streaming token output are the next runtime layer.
- Keyword memory and skill retrieval are intentionally simple. Embedding retrieval can be added behind the same interfaces.
- The Transformers backend is generic, but some model repositories require model-specific preprocessing code.
- Skill validation exposes the correct generate-validate-promote boundary, while benchmark-specific replay policies still need to be supplied by the experiment.

## Repository layout

```text
src/dualstream_agent/
├── backends/      # mock, OpenAI-compatible, Transformers
├── perception/    # frame buffer, change detector, causal state
├── systems/       # S1, controller, S2
├── runtime/       # session, stream sources, trace store
├── memory/        # SQLite episodic memory
├── skills/        # SKILL.md loading and retrieval
├── tools/         # bounded workspace tools
├── evolution/     # candidate generation and replay validation
└── harness/       # scenarios, runner, metrics, checkers
```

## Safety

The default workspace tools reject paths outside the configured workspace. Scenario command checkers execute shell commands, so only run trusted scenario packages or isolate them in a container.
