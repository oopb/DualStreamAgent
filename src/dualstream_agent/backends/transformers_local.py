from __future__ import annotations

import asyncio
import io
import time
from typing import Any

from PIL import Image

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.schemas import GenerationRequest, GenerationResult


def _as_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.fromarray(image).convert("RGB")


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
        common = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        # Transformers 5 uses dtype; current 4.x accepts torch_dtype. Try the
        # modern spelling first and fall back for compatibility.
        common["dtype"] = dtype

        if multimodal:
            self.processor = transformers.AutoProcessor.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_cls is None:
                model_cls = transformers.AutoModelForVision2Seq
            try:
                self.model = model_cls.from_pretrained(model, **common)
            except TypeError:
                common["torch_dtype"] = common.pop("dtype")
                self.model = model_cls.from_pretrained(model, **common)
            self.tokenizer = None
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            try:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **common)
            except TypeError:
                common["torch_dtype"] = common.pop("dtype")
                self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **common)
            self.processor = None
        self.model.eval()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        with self.torch.inference_mode():
            if self.multimodal:
                inputs = self._prepare_multimodal(request)
                input_length = int(inputs["input_ids"].shape[-1])
            else:
                inputs = self._prepare_text(request)
                input_length = int(inputs["input_ids"].shape[-1])

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "do_sample": request.temperature > 0,
            }
            if request.temperature > 0:
                generate_kwargs["temperature"] = request.temperature
            outputs = self.model.generate(**inputs, **generate_kwargs)
            generated = outputs[:, input_length:]
            decoder = self.processor if self.multimodal else self.tokenizer
            text = decoder.batch_decode(generated, skip_special_tokens=True)[0].strip()

        return GenerationResult(
            text=text,
            prompt_tokens=input_length,
            completion_tokens=int(generated.shape[-1]),
            latency_s=time.perf_counter() - started,
        )

    def _prepare_text(self, request: GenerationRequest) -> dict[str, Any]:
        assert self.tokenizer is not None
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in request.messages
            )
            inputs = self.tokenizer(text, return_tensors="pt")
        return self._move_inputs(inputs)

    def _prepare_multimodal(self, request: GenerationRequest) -> dict[str, Any]:
        assert self.processor is not None
        messages = [dict(message) for message in request.messages]
        images = [_as_pil(image) for image in request.images]
        if images:
            user_index = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
                None,
            )
            if user_index is None:
                messages.append({"role": "user", "content": ""})
                user_index = len(messages) - 1
            original = messages[user_index].get("content", "")
            content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": str(original)})
            messages[user_index]["content"] = content

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            # Compatibility fallback for processors that expect text and images
            # as separate arguments.
            text_messages = []
            for message in request.messages:
                text_messages.append(
                    {"role": message.get("role", "user"), "content": str(message.get("content", ""))}
                )
            prompt = self.processor.apply_chat_template(
                text_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=prompt, images=images or None, return_tensors="pt")
        return self._move_inputs(inputs)

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        target = next(self.model.parameters()).device
        moved: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            moved[key] = value.to(target) if hasattr(value, "to") else value
        return moved
