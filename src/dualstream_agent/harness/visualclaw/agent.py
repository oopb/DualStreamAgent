from __future__ import annotations

from typing import Any

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.backends.factory import build_backend
from dualstream_agent.config import AppConfig
from dualstream_agent.schemas import GenerationRequest

from .prompt import SYSTEM_PROMPT
from .schemas import AgentTurn, VisualClawRound


class VisualClawBaselineAgent:
    """Single-call VLM baseline using DualStreamAgent's local model backends.

    This intentionally bypasses S1 response gating: VisualClawArena already
    provides an explicit user instruction for every round. The runner is a
    stable baseline harness; later dual-system experiments can replace this
    class without changing dataset loading, staging, updates, or scoring.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        backend: ModelBackend | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.backend = backend or build_backend(config.s2)
        self._owns_backend = backend is None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    async def run_round(
        self,
        round_item: VisualClawRound,
        *,
        prompt: str,
        images: list[Any],
        dry_run: bool = False,
    ) -> AgentTurn:
        if dry_run:
            if round_item.round_type == "multi_choice":
                expected = round_item.evaluation.get("answer", ["A"])
                if isinstance(expected, str):
                    answer = expected
                elif isinstance(expected, list) and expected:
                    answer = str(expected[0])
                else:
                    answer = "A"
                return AgentTurn(response=f"(dry-run oracle smoke test)\n\n\\bbox{{{answer}}}")
            return AgentTurn(response="(dry-run: exec_check does not synthesize files)")

        result = await self.backend.generate(
            GenerationRequest(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                images=images,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        )
        return AgentTurn(
            response=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=result.latency_s,
            raw=result.raw,
        )

    async def close(self) -> None:
        if self._owns_backend:
            await self.backend.close()
