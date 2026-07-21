import httpx


class OpenAICompatibleBackend:
    """Adapter for local vLLM/SGLang/llama.cpp OpenAI-compatible servers."""

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    async def generate(self, messages, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    **kwargs,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
