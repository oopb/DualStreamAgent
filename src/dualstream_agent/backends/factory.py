from __future__ import annotations

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.backends.mock import MockBackend
from dualstream_agent.backends.openai_compatible import OpenAICompatibleBackend
from dualstream_agent.backends.transformers_local import TransformersBackend
from dualstream_agent.config import BackendConfig


def build_backend(config: BackendConfig) -> ModelBackend:
    if config.kind == "mock":
        return MockBackend()
    if config.kind == "openai_compatible":
        return OpenAICompatibleBackend(
            endpoint=config.endpoint,
            model=config.model,
            api_key=config.api_key,
        )
    if config.kind == "transformers":
        return TransformersBackend(
            model=config.model,
            multimodal=config.multimodal,
            device_map=config.device_map,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
    raise ValueError(f"Unsupported backend kind: {config.kind}")
