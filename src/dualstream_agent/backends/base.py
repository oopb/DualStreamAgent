from __future__ import annotations

from abc import ABC, abstractmethod

from dualstream_agent.schemas import GenerationRequest, GenerationResult


class ModelBackend(ABC):
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None
