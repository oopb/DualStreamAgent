# VisualClawArena compatibility

DualStreamAgent includes a dedicated adapter for the public
`UCSC-VLAA/VisualClawArena` Hugging Face release. The adapter is intentionally
separate from the live S1 response gate: VisualClawArena supplies an explicit
instruction on every round, so the compatibility baseline calls the configured
S2/local VLM once per round. Dataset loading, workspace state, updates, video
frames, logging, and scoring can later be reused by dual-system agents.

## Supported release protocol

- `manifest.json` and `manifests/scenarios.jsonl` discovery.
- Hugging Face layout: `scenarios/<scenario_id>/{data,spec}`.
- `spec/questions.json` multi-round loading.
- Release, deprecated, video-required, round-type, and round-id filters.
- One isolated work copy per scenario run.
- Persistent workspace and session history across rounds.
- Deferred `update_ids` with new/replace/delete/append/directory operations.
- Uniform walkthrough-frame extraction and staging under `_clip_frames/`.
- Explicit question-aware `cited` and `uniform+cited` ablations.
- Multiple-choice extraction compatible with `\\bbox{X}`.
- Per-round exec-check commands with `${eval_dir}`, `${agent_id}`, and
  `${workspace}` expansion.
- `### WRITE_FILE:` blocks for local VLMs without native file tools.
- Per-round prompts, answers, scores, update records, result JSON, and aggregate
  JSONL/summary output.

The runner never copies `spec/`, checker scripts, gold files, future questions,
or construction notes into the agent workspace. Only the current round is added
to the prompt.

## Installation

Core tests and mock backend:

```bash
pip install -e '.[dev]'
```

Dataset download, video extraction, and common checker dependencies:

```bash
pip install -e '.[arena,dev]'
```

For an in-process Hugging Face model:

```bash
pip install -e '.[arena,transformers,dev]'
```

## Download

```bash
dualstream-visualclaw download ./data/VisualClawArena
```

An existing `git lfs` or `huggingface-cli` checkout can be used directly as
long as its root contains `manifest.json`, `manifests/`, and `scenarios/`.
Dataset media is not copied into this repository.

## Validate the release

```bash
dualstream-visualclaw validate ./data/VisualClawArena
```

A quick structural check:

```bash
dualstream-visualclaw validate ./data/VisualClawArena --max-scenarios 5
```

## Harness smoke test

This tests loading, workspace staging, update timing, MCQ scoring, logging, and
result export without loading a model. `exec_check` rounds normally fail in
`--dry-run` because dry-run does not fabricate output files.

```bash
dualstream-visualclaw run ./data/VisualClawArena \
  --config configs/mock.yaml \
  --scenario <scenario_id> \
  --max-rounds 3 \
  --keyframe-mode none \
  --allow-missing-clips \
  --dry-run
```

To validate only MCQ wiring:

```bash
dualstream-visualclaw run ./data/VisualClawArena \
  --config configs/mock.yaml \
  --scenario <scenario_id> \
  --round-type multi_choice \
  --max-rounds 3 \
  --keyframe-mode none \
  --dry-run
```

## Run a local Transformers VLM

Configure `configs/transformers.yaml`, then run:

```bash
dualstream-visualclaw run ./data/VisualClawArena \
  --config configs/transformers.yaml \
  --scenario <scenario_id> \
  --max-keyframes 8 \
  --keyframe-mode uniform \
  --checker-mode host
```

Run several scenarios:

```bash
dualstream-visualclaw run ./data/VisualClawArena \
  --config configs/transformers.yaml \
  --scenario <scenario_a> \
  --scenario <scenario_b> \
  --output runs/visualclaw
```

Run the complete release:

```bash
dualstream-visualclaw run ./data/VisualClawArena \
  --config configs/transformers.yaml \
  --all \
  --output runs/visualclaw \
  --resume
```

`uniform` is the default and does not use question timestamps to select frames.
`cited` and `uniform+cited` are question-aware ablations and should be reported
as such; they are not strict causal streaming settings.

## Local OpenAI-compatible server

The same runner supports vLLM, SGLang, llama.cpp server, or another compatible
local endpoint through `configs/openai-compatible.yaml`:

```bash
dualstream-visualclaw run ./data/VisualClawArena \
  --config configs/openai-compatible.yaml \
  --scenario <scenario_id>
```

## Output layout

```text
runs/visualclaw/
├── <scenario_id>/<run_id>/
│   ├── workspace/                 # persistent writable copy
│   │   └── _clip_frames/
│   ├── sessions/                  # staged session history
│   ├── agent_logs/<round_id>/
│   │   ├── prompt.txt
│   │   ├── answer.txt
│   │   └── result.json
│   ├── access_logs/rounds.jsonl
│   └── results.json
├── summary-<run_id>.json
└── per-round-<run_id>.jsonl
```

## Checker safety

`--checker-mode host` executes the dataset's checker command as a subprocess.
It uses argument parsing rather than `shell=True`, replaces bare `python` with
the runner's interpreter, applies the declared timeout, and runs with the
staged workspace as the current directory. It is still executable benchmark
code. Run trusted releases in a disposable environment or container.

Disable checkers during parser/prompt debugging:

```bash
--checker-mode disabled
```

Disabled exec-check rounds are marked skipped, not passed.

## Baseline and future research modes

The current VisualClawArena compatibility runner is a stable single-call
baseline:

```text
current round + current workspace + sessions + selected frames
    -> configured S2/local model backend
    -> response / WRITE_FILE blocks
    -> official-style scorer
```

It intentionally does not force an explicit-user task through the live
silence/response gate. Future experiments should add alternate agent classes
behind the same runner, for example:

- S1-only agent.
- S1-to-S2 routed agent.
- Memory-enabled agent.
- Skill-enabled agent.
- Evolved-skill agent.
- Strict causal video replay rather than static keyframes.

Keeping those changes behind the agent interface preserves identical scenario
loading, updates, workspaces, and scorers for fair ablations.
