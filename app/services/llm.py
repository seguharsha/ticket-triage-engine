"""One LLMClient interface, several providers; switching is a config change.

The LLM is only used for low-confidence classification and reply drafts. Rate limits (429) and
transient errors raise, and the Celery task retries them with backoff like any other failure.
"""

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import httpx

from app.config import settings


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * settings.llm_price_input_per_mtok
            + self.output_tokens * settings.llm_price_output_per_mtok
        ) / 1_000_000


class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str, *, max_tokens: int = 512, json_mode: bool = False) -> LLMResult: ...


def _raise_for_status(resp: httpx.Response, provider: str) -> None:
    if resp.status_code == 429:
        raise RateLimited(f"{provider} rate limited: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise LLMError(f"{provider} HTTP {resp.status_code}: {resp.text[:300]}")


class GroqClient:
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: float, transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.model = f"groq/{model}"
        self._model = model
        self.http = httpx.Client(timeout=timeout, transport=transport, headers={"Authorization": f"Bearer {api_key}"})

    def complete(self, prompt: str, *, max_tokens: int = 512, json_mode: bool = False) -> LLMResult:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        resp = self.http.post(self.url, json=body)
        _raise_for_status(resp, "groq")
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class OllamaClient:
    """Local model via Ollama.

    Prompts put fixed instructions first and the ticket last, so Ollama can reuse the cached
    prefix between calls.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = f"ollama/{model}"
        self._model = model
        self.num_ctx, self.keep_alive = num_ctx, keep_alive
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def complete(self, prompt: str, *, max_tokens: int = 512, json_mode: bool = False) -> LLMResult:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        if json_mode:
            body["format"] = "json"
        start = time.perf_counter()
        try:
            resp = self.http.post("/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise LLMError(f"cannot reach Ollama at {self.http.base_url}; is it running?") from exc
        if resp.status_code == 404 and "not found" in resp.text:
            raise LLMError(f"model {self._model!r} not pulled; run: ollama pull {self._model}")
        _raise_for_status(resp, "ollama")
        data = resp.json()
        return LLMResult(
            text=data["message"]["content"],
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class FakeLLMClient:
    """Fixed answers after a simulated delay, so load tests measure this system, not a provider."""

    model = "fake"

    def __init__(self, delay_ms: int = 0):
        self.delay_ms = delay_ms
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 512, json_mode: bool = False) -> LLMResult:
        self.prompts.append(prompt)
        time.sleep(self.delay_ms / 1000)
        if json_mode:
            text = json.dumps({"category": "bug", "priority": "P3", "team": "Product"})
        else:
            text = "Thanks for reaching out. A similar issue was resolved by clearing the cache and signing in again [S1]."
        return LLMResult(text, self.model, len(prompt) // 4, len(text) // 4, float(self.delay_ms))


@lru_cache
def get_llm_client() -> LLMClient | None:
    provider = settings.llm_provider
    if provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model, settings.llm_timeout_seconds)
    if provider == "ollama":
        return OllamaClient(settings.ollama_url, settings.ollama_model, settings.llm_timeout_seconds,
                            settings.ollama_num_ctx, settings.ollama_keep_alive)
    if provider == "fake":
        return FakeLLMClient(settings.fake_llm_delay_ms)
    return None


def extract_json(text: str) -> dict | None:
    """Parse a JSON object even if the model wrapped it in prose or a code fence."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
