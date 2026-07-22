from __future__ import annotations

import asyncio
import time
from typing import Any

from dualstream_agent.backends.adapters import (
    ModelAdapter,
    MultimodalModelAdapter,
    TextModelAdapter,
)
from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.schemas import GenerationRequest, GenerationResult


class TransformersBackend(ModelBackend):
    """In-process Hugging Face Transformers backend.

    Imports are lazy so the core package remains lightweight. Text-only models
    use AutoModelForCausalLM. Multimodal models prefer
    AutoModelForImageTextToText and fall back to AutoModelForVision2Seq for
    older Transformers releases.
    """

    def __init__(
        self,
        model: str,
        *,
        multimodal: bool = False,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "Install the local backend with: pip install -e '.[transformers]'"
            ) from exc

        self.torch = torch
        self.transformers = transformers
        self.model_name = model
        self.multimodal = multimodal
        major_version = int(transformers.__version__.split(".", 1)[0])
        common = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "dtype" if major_version >= 5 else "torch_dtype": dtype,
        }

        if multimodal:
            self.processor = transformers.AutoProcessor.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_cls is None:
                model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
            if model_cls is None:
                model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_cls is None:
                raise RuntimeError("This Transformers version has no multimodal auto model class")
            self.model = model_cls.from_pretrained(model, **common)
            self.tokenizer = None
            self.adapter: ModelAdapter = MultimodalModelAdapter(self.processor)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **common)
            self.processor = None
            self.adapter = TextModelAdapter(self.tokenizer)
        self.model.eval()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.timeout_s is not None and request.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        work = asyncio.to_thread(self._generate_sync, request)
        if request.timeout_s is None:
            return await work
        try:
            return await asyncio.wait_for(work, timeout=request.timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Transformers generation timed out after {request.timeout_s}s"
            ) from exc

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                inputs = self._move_inputs(self.adapter.prepare_inputs(request))
                input_length = int(inputs["input_ids"].shape[-1])
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = self.torch.ones_like(inputs["input_ids"])

                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": request.max_new_tokens,
                    "do_sample": request.temperature > 0,
                }
                if request.temperature > 0:
                    generate_kwargs["temperature"] = request.temperature
                if self.adapter.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = self.adapter.pad_token_id
                outputs = self.model.generate(**inputs, **generate_kwargs)
                if getattr(self.model.config, "is_encoder_decoder", False):
                    generated = outputs
                else:
                    generated = outputs[:, input_length:]
                text = self.adapter.decode_output(generated)
                text = self._apply_stop(text, request.stop)
        except Exception as exc:
            raise RuntimeError(
                f"Transformers generation failed for {self.model_name}: {exc}"
            ) from exc

        return GenerationResult(
            text=text,
            prompt_tokens=input_length,
            completion_tokens=int(generated.shape[-1]),
            latency_s=time.perf_counter() - started,
        )

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        target = getattr(self.model, "device", None)
        if target is None or getattr(target, "type", None) == "meta":
            target = next(
                (
                    parameter.device
                    for parameter in self.model.parameters()
                    if getattr(parameter.device, "type", None) != "meta"
                ),
                None,
            )
        moved: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            moved[key] = value.to(target) if target is not None and hasattr(value, "to") else value
        return moved

    @staticmethod
    def _apply_stop(text: str, stop: list[str] | None) -> str:
        positions = [text.find(token) for token in stop or [] if token and token in text]
        return text[: min(positions)].rstrip() if positions else text
