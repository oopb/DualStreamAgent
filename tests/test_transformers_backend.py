from __future__ import annotations

import asyncio
import contextlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dualstream_agent.backends.adapters import MultimodalModelAdapter, TextModelAdapter
from dualstream_agent.backends.transformers_local import TransformersBackend
from dualstream_agent.schemas import GenerationRequest


class _TokenizerWithoutTemplate:
    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self):
        self.prompt = ""
        self.kwargs = {}

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("no chat template")

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return {"input_ids": np.array([[1, 2]])}

    def batch_decode(self, generated, **kwargs):
        return ["decoded"]


def test_text_adapter_falls_back_when_chat_template_is_missing():
    tokenizer = _TokenizerWithoutTemplate()
    adapter = TextModelAdapter(tokenizer)

    inputs = adapter.prepare_inputs(
        GenerationRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert inputs["input_ids"].shape == (1, 2)
    assert tokenizer.prompt == "user: hello"
    assert tokenizer.kwargs == {"return_tensors": "pt"}


class _RecordingProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token="<eos>")

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }

    def batch_decode(self, generated, **kwargs):
        return ["decoded"]


def test_multimodal_adapter_isolates_image_message_format():
    processor = _RecordingProcessor()
    adapter = MultimodalModelAdapter(processor)
    messages = [{"role": "user", "content": "describe"}]

    adapter.prepare_inputs(
        GenerationRequest(
            messages=messages,
            images=[Image.new("RGB", (2, 2), "red")],
        )
    )

    assert messages == [{"role": "user", "content": "describe"}]
    blocks = processor.messages[0]["content"]
    assert [block["type"] for block in blocks] == ["image", "text"]
    assert isinstance(blocks[0]["image"], Image.Image)


class _ProcessorWithoutTemplate:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token="<eos>")
    image_token = "<image>"

    def __init__(self):
        self.prompt = ""

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("no chat template")

    def __call__(self, *, text, images, return_tensors):
        self.prompt = text
        return {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }

    def batch_decode(self, generated, **kwargs):
        return ["decoded"]


def test_multimodal_adapter_adds_image_tokens_without_chat_template():
    processor = _ProcessorWithoutTemplate()
    adapter = MultimodalModelAdapter(processor)

    adapter.prepare_inputs(
        GenerationRequest(
            messages=[{"role": "user", "content": "describe"}],
            images=[Image.new("RGB", (2, 2), "red")],
        )
    )

    assert processor.prompt == "<image>\nuser: describe"


class _FakeAdapter:
    pad_token_id = 7

    def prepare_inputs(self, request):
        return {"input_ids": np.array([[10, 11]])}

    def decode_output(self, generated):
        assert generated.tolist() == [[12, 13]]
        return "answer<stop>ignored"


class _FakeModel:
    device = "cpu"
    config = SimpleNamespace(is_encoder_decoder=False)

    def generate(self, **kwargs):
        assert kwargs["attention_mask"].tolist() == [[1, 1]]
        assert kwargs["pad_token_id"] == 7
        assert kwargs["do_sample"] is False
        return np.array([[10, 11, 12, 13]])


def test_transformers_generation_slices_prompt_and_applies_stop():
    backend = object.__new__(TransformersBackend)
    backend.model_name = "fake"
    backend.multimodal = False
    backend.model = _FakeModel()
    backend.adapter = _FakeAdapter()
    backend.torch = SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        ones_like=np.ones_like,
    )

    result = backend._generate_sync(
        GenerationRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_new_tokens=2,
            stop=["<stop>"],
        )
    )

    assert result.text == "answer"
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 2


@pytest.mark.asyncio
async def test_transformers_generation_timeout(monkeypatch):
    async def slow_to_thread(*args, **kwargs):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(asyncio, "to_thread", slow_to_thread)
    backend = object.__new__(TransformersBackend)

    with pytest.raises(TimeoutError, match="timed out"):
        await backend.generate(
            GenerationRequest(messages=[], timeout_s=0.001)
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("DUALSTREAM_RUN_TRANSFORMERS_INTEGRATION") != "1",
    reason="set DUALSTREAM_RUN_TRANSFORMERS_INTEGRATION=1",
)
def test_real_tiny_text_model():
    backend = TransformersBackend(
        os.getenv("DUALSTREAM_TEXT_MODEL", "sshleifer/tiny-gpt2"),
        device_map="cpu",
        dtype="float32",
        local_files_only=os.getenv("DUALSTREAM_LOCAL_FILES_ONLY") == "1",
    )
    result = backend._generate_sync(
        GenerationRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_new_tokens=2,
        )
    )
    assert result.prompt_tokens and result.prompt_tokens > 0
    assert result.completion_tokens == 2


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("DUALSTREAM_RUN_VLM_INTEGRATION") != "1",
    reason="set DUALSTREAM_RUN_VLM_INTEGRATION=1",
)
def test_real_tiny_vlm():
    backend = TransformersBackend(
        os.getenv(
            "DUALSTREAM_VLM_MODEL",
            "Xenova/tiny-random-LlavaForConditionalGeneration",
        ),
        multimodal=True,
        device_map="cpu",
        dtype="float32",
        local_files_only=os.getenv("DUALSTREAM_LOCAL_FILES_ONLY") == "1",
    )
    result = backend._generate_sync(
        GenerationRequest(
            messages=[{"role": "user", "content": "Describe the image."}],
            images=[Image.new("RGB", (32, 32), "red")],
            max_new_tokens=1,
        )
    )
    assert result.prompt_tokens and result.prompt_tokens > 0
    assert result.completion_tokens == 1
