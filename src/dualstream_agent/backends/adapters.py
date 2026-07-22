from __future__ import annotations

import copy
import io
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from dualstream_agent.schemas import GenerationRequest


def _as_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        with Image.open(image) as loaded:
            return loaded.convert("RGB")
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as loaded:
            return loaded.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content or "")


def _plain_prompt(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{message.get('role', 'user')}: {_content_text(message.get('content', ''))}"
        for message in messages
    )


class ModelAdapter(ABC):
    """Isolate tokenizer and processor differences from generation."""

    @abstractmethod
    def prepare_inputs(self, request: GenerationRequest) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode_output(self, generated: Any) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        raise NotImplementedError


class TextModelAdapter(ModelAdapter):
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token

    def prepare_inputs(self, request: GenerationRequest) -> Any:
        try:
            prompt = self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        except (AttributeError, TypeError, ValueError):
            return self.tokenizer(_plain_prompt(request.messages), return_tensors="pt")

    def decode_output(self, generated: Any) -> str:
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self.tokenizer, "pad_token_id", None)


class MultimodalModelAdapter(ModelAdapter):
    """Generic adapter for processors implementing multimodal chat templates."""

    def __init__(self, processor: Any):
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token

    def prepare_inputs(self, request: GenerationRequest) -> Any:
        images = self.prepare_images(request)
        messages = self.prepare_messages(request, images=images)
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            text_messages = [
                {
                    "role": message.get("role", "user"),
                    "content": _content_text(message.get("content", "")),
                }
                for message in request.messages
            ]
            try:
                prompt = self.processor.apply_chat_template(
                    text_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, TypeError, ValueError):
                prompt = _plain_prompt(text_messages)
                image_token = getattr(self.processor, "image_token", None)
                if images and image_token:
                    prompt = f"{image_token * len(images)}\n{prompt}"
            return self.processor(text=prompt, images=images or None, return_tensors="pt")

    def prepare_messages(
        self,
        request: GenerationRequest,
        *,
        images: list[Image.Image] | None = None,
    ) -> list[dict[str, Any]]:
        messages = copy.deepcopy(request.messages)
        images = self.prepare_images(request) if images is None else images
        if not images:
            return messages
        user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        if user_index is None:
            messages.append({"role": "user", "content": []})
            user_index = len(messages) - 1
        original = messages[user_index].get("content", "")
        original_blocks = list(original) if isinstance(original, list) else []
        if not original_blocks and original:
            original_blocks.append({"type": "text", "text": str(original)})
        messages[user_index]["content"] = [
            *({"type": "image", "image": image} for image in images),
            *original_blocks,
        ]
        return messages

    @staticmethod
    def prepare_images(request: GenerationRequest) -> list[Image.Image]:
        return [_as_pil(image) for image in request.images]

    def decode_output(self, generated: Any) -> str:
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @property
    def pad_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        return getattr(tokenizer, "pad_token_id", None)
