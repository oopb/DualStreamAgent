from __future__ import annotations

import base64
import io
import time
from typing import Any

import httpx
from PIL import Image

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.schemas import GenerationRequest, GenerationResult


def _image_to_data_url(image: Any) -> str:
    if isinstance(image, (str, bytes)):
        if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
            return image
        if isinstance(image, str):
            image = Image.open(image)
        else:
            image = Image.open(io.BytesIO(image))
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class OpenAICompatibleBackend(ModelBackend):
    """Adapter for local vLLM, SGLang, llama.cpp, and compatible servers."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 180.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        messages = [dict(message) for message in request.messages]
        if request.images:
            user_index = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
                None,
            )
            if user_index is None:
                messages.append({"role": "user", "content": ""})
                user_index = len(messages) - 1
            original = messages[user_index].get("content", "")
            content: list[dict[str, Any]] = [{"type": "text", "text": str(original)}]
            content.extend(
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}}
                for image in request.images
            )
            messages[user_index]["content"] = content

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            payload["stop"] = request.stop
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "dualstream_response",
                    "schema": request.response_schema,
                },
            }

        started = time.perf_counter()
        response = await self.client.post(f"{self.endpoint}/chat/completions", json=payload)
        if response.status_code == 400 and "response_format" in payload:
            # Several local OpenAI-compatible servers do not implement JSON
            # schema mode. Retry once and rely on prompt-level JSON guidance.
            payload.pop("response_format", None)
            response = await self.client.post(f"{self.endpoint}/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        return GenerationResult(
            text=body["choices"][0]["message"].get("content", ""),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=time.perf_counter() - started,
            raw=body,
        )

    async def close(self) -> None:
        await self.client.aclose()
