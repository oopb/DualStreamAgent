---
name: observe_scene
description: Decide whether a visual change is relevant enough to mention.
version: 1
status: active
tags:
  - streaming
  - response-timing
  - visual-observation
---

## Applicability

Use during passive or task-directed visual observation.

## Procedure

1. Separate pixel-level change from task-relevant semantic change.
2. Check whether the event is new relative to the recent causal state.
3. Prefer silence for repeated, irrelevant, or low-confidence observations.
4. Respond immediately only for urgent events or a clearly satisfied user request.
5. Route to System 2 when long-term recall, verification, planning, or a tool is required.

## Verification

Before speaking, identify the concrete new evidence and why it matters now.

## Anti-patterns

Do not narrate every frame. Do not use future frames or future questions.
