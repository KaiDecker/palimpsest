"""Generator abstraction.

The default generator is deterministic so the project works offline. A caller
can inject any object implementing ``generate`` into ``create_app``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Protocol
from urllib.parse import urlparse


class Generator(Protocol):
    model_name: str
    adapter: str | None

    def generate(
        self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]
    ) -> str: ...

    def generate_stream(
        self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]
    ) -> Iterator[str]: ...


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

    def generate_stream(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> Iterator[str]:
        """Yield a single chunk so the offline generator follows the stream contract."""
        yield self.generate(prompt, history, memories, profile)


class OpenAICompatibleGenerator:
    """Minimal client for local OpenAI-compatible chat-completion servers.

    It uses only the Python standard library, so llama.cpp, vLLM, and LM Studio
    can be used without adding a hosted-provider SDK or a hard network dependency.
    """

    adapter = None

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Model endpoint must be an absolute http:// or https:// URL")
        if timeout <= 0:
            raise ValueError("Model endpoint timeout must be greater than zero")
        self.base_url = normalized_url
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
        timeout_value = os.getenv("PALIMPSEST_MODEL_TIMEOUT", "120")
        try:
            timeout = float(timeout_value)
        except ValueError:
            timeout = 120.0
        return cls(base_url, model_name, os.getenv("PALIMPSEST_API_KEY") or os.getenv("OPENAI_API_KEY"), timeout)

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

    @property
    def models_endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url[: -len("/chat/completions")] + "/models"
        if self.base_url.endswith("/models"):
            return self.base_url
        return f"{self.base_url}/models"

    def _messages(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> list[dict[str, str]]:
        system_parts = ["You are Palimpsest, a local-first personal assistant."]
        if memories:
            system_parts.append("Relevant memories: " + "; ".join(item["content"] for item in memories[:5]))
        if profile:
            system_parts.append("User profile: " + "; ".join(f"{item['key']}={item['value']}" for item in profile[:10]))
        clean_history = [{"role": item["role"], "content": item["content"]} for item in history if item.get("role") in {"user", "assistant", "system"}]
        return [{"role": "system", "content": "\n".join(system_parts)}, *clean_history, {"role": "user", "content": prompt}]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _error_detail(exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
            return f"HTTP {exc.code}" + (f": {detail[:300]}" if detail else "")
        if isinstance(exc, urllib.error.URLError):
            reason = exc.reason
            return f"连接失败：{reason}"
        if isinstance(exc, TimeoutError):
            return "请求超时"
        return str(exc)

    @staticmethod
    def _content_from_payload(data: Any) -> str:
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None:
                content = (choice.get("delta") or {}).get("content")
            if content is None:
                content = choice.get("text")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise GenerationError("Local model endpoint returned an invalid chat-completion response") from exc
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        if not isinstance(content, str):
            raise GenerationError("Local model endpoint returned an invalid response content")
        return content

    def generate(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> str:
        payload = {"model": self.model_name, "messages": self._messages(prompt, history, memories, profile), "stream": False}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Local model endpoint failed: {self._error_detail(exc)}") from exc
        content = self._content_from_payload(data)
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Local model endpoint returned an empty response")
        return content.strip()

    def generate_stream(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> Iterator[str]:
        """Yield text deltas from an OpenAI-compatible SSE or JSON response.

        Some local servers ignore ``stream`` and return one JSON document. That
        response is accepted and yielded as one chunk for graceful fallback.
        """
        payload = {"model": self.model_name, "messages": self._messages(prompt, history, memories, profile), "stream": True}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers={**self._headers(), "Accept": "text/event-stream, application/json"}, method="POST")
        emitted = False
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # A server may split an SSE event across physical lines;
                        # incomplete lines are ignored rather than crashing the UI.
                        continue
                    try:
                        chunk = self._content_from_payload(data)
                    except GenerationError:
                        continue
                    if chunk:
                        emitted = True
                        yield chunk
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GenerationError(f"Local model endpoint failed: {self._error_detail(exc)}") from exc
        if not emitted:
            raise GenerationError("Local model endpoint returned no generated content")

    def diagnose(self) -> dict[str, Any]:
        """Probe the standard ``/models`` endpoint without raising to callers."""
        started = time.perf_counter()
        request = urllib.request.Request(self.models_endpoint, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 5.0)) as response:
                data = json.loads(response.read().decode("utf-8"))
            raw_models = data.get("data", []) if isinstance(data, dict) else []
            models = [item.get("id") for item in raw_models if isinstance(item, dict) and item.get("id")]
            return {"status": "connected", "reachable": True, "endpoint": self.base_url, "models_endpoint": self.models_endpoint, "model": self.model_name, "models": models, "latency_ms": round((time.perf_counter() - started) * 1000, 1), "error": None}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return {"status": "error", "reachable": False, "endpoint": self.base_url, "models_endpoint": self.models_endpoint, "model": self.model_name, "models": [], "latency_ms": round((time.perf_counter() - started) * 1000, 1), "error": self._error_detail(exc)}

    def generate_variant(self, prompt: str, history: list[dict[str, str]], memories: list[dict[str, Any]], profile: list[dict[str, Any]]) -> str:
        variant_prompt = prompt + "\n\nProvide a meaningfully different alternative answer from your first candidate."
        return self.generate(variant_prompt, history, memories, profile)
