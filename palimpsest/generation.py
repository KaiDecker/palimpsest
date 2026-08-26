"""Generator abstraction.

The default generator is deterministic so the project works offline. A caller
can inject any object implementing ``generate`` into ``create_app``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol


class Generator(Protocol):
    model_name: str
    adapter: str | None

    def generate(
        self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]
    ) -> str: ...


class GenerationError(RuntimeError):
    """Raised when a configured model endpoint cannot produce a response."""


class MockGenerator:
    model_name = "palimpsest-mock-v1"
    adapter = None

    def generate(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> str:
        # Keep this stable for tests and useful as a smoke-test response.
        answer = f"I hear you: {prompt.strip()}"
        if memories:
            memory_text = "; ".join(item["content"] for item in memories[:3])
            answer += f"\n\nRelevant memory: {memory_text}"
        if any(item.get("key") == "preference.communication" for item in profile):
            answer += "\n\nI’ll keep this response focused and practical."
        return answer

    def generate_variant(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> str:
        """Produce a deterministic second candidate for A/B feedback."""
        return f"Alternative take: {prompt.strip()}\n\nI’ll answer this from a different angle while staying concise."


class OpenAICompatibleGenerator:
    """Minimal client for local OpenAI-compatible chat-completion servers.

    It uses only the Python standard library, so llama.cpp, vLLM, and LM Studio
    can be used without adding a hosted-provider SDK or a hard network dependency.
    """

    adapter = None

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleGenerator | None":
        base_url = (
            os.getenv("PALIMPSEST_MODEL_ENDPOINT")
            or os.getenv("PALIMPSEST_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        if not base_url:
            return None
        model_name = os.getenv("PALIMPSEST_MODEL_NAME") or os.getenv("PALIMPSEST_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "local-model"
        return cls(base_url, model_name, os.getenv("PALIMPSEST_API_KEY") or os.getenv("OPENAI_API_KEY"))

    @classmethod
    def model_names_from_env(cls, default: str) -> list[str]:
        configured = os.getenv("PALIMPSEST_MODEL_NAMES", "")
        names = [name.strip() for name in configured.split(",") if name.strip()]
        return names or [default]

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _messages(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> list[dict[str, str]]:
        system_parts = ["You are Palimpsest, a local-first personal assistant."]
        if memories:
            system_parts.append("Relevant memories: " + "; ".join(item["content"] for item in memories[:5]))
        if profile:
            system_parts.append("User profile: " + "; ".join(f"{item['key']}={item['value']}" for item in profile[:10]))
        clean_history = [{"role": item["role"], "content": item["content"]} for item in history if item.get("role") in {"user", "assistant", "system"}]
        return [{"role": "system", "content": "\n".join(system_parts)}, *clean_history, {"role": "user", "content": prompt}]

    def generate(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> str:
        payload = {"model": self.model_name, "messages": self._messages(prompt, history, memories, profile), "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Local model endpoint failed: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError("Local model endpoint returned an invalid chat-completion response") from exc
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Local model endpoint returned an empty response")
        return content.strip()

    def generate_variant(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> str:
        variant_prompt = prompt + "\n\nProvide a meaningfully different alternative answer from your first candidate."
        return self.generate(variant_prompt, history, memories, profile)
