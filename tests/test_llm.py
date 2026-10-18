import json

import httpx
import pytest

from app.services.llm import FakeLLMClient, GroqClient, LLMError, OllamaClient, RateLimited, extract_json


def transport(status=200, payload=None, capture=None):
    def handler(request: httpx.Request):
        if capture is not None:
            capture.append(json.loads(request.content))
        return httpx.Response(status, json=payload or {})
    return httpx.MockTransport(handler)


def test_groq_parses_text_and_usage():
    sent = []
    payload = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    client = GroqClient("k", "llama", 5, transport=transport(payload=payload, capture=sent))
    result = client.complete("hello", json_mode=True)
    assert (result.text, result.input_tokens, result.output_tokens, result.model) == ("hi", 10, 2, "groq/llama")
    assert sent[0]["response_format"] == {"type": "json_object"} and sent[0]["temperature"] == 0


def test_groq_429_raises_rate_limited():
    client = GroqClient("k", "llama", 5, transport=transport(status=429))
    with pytest.raises(RateLimited):
        client.complete("x")


def test_groq_server_error_raises():
    client = GroqClient("k", "llama", 5, transport=transport(status=500))
    with pytest.raises(LLMError):
        client.complete("x")


def test_groq_requires_key():
    with pytest.raises(LLMError):
        GroqClient("", "llama", 5)


def test_ollama_parses_response():
    sent = []
    payload = {"message": {"content": "ok"}, "prompt_eval_count": 7, "eval_count": 3}
    client = OllamaClient("http://ollama", "llama3", 5, transport=transport(payload=payload, capture=sent))
    result = client.complete("hello", json_mode=True)
    assert (result.text, result.input_tokens, result.output_tokens) == ("ok", 7, 3)
    assert sent[0]["format"] == "json" and sent[0]["stream"] is False
    assert sent[0]["options"]["num_ctx"] == 4096 and sent[0]["keep_alive"] == "30m"


def test_ollama_not_running_gives_clear_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    client = OllamaClient("http://ollama", "llama3", 5, transport=httpx.MockTransport(refuse))
    with pytest.raises(LLMError, match="is it running"):
        client.complete("x")


def test_ollama_model_not_pulled_gives_clear_error():
    def missing(request):
        return httpx.Response(404, json={"error": "model 'llama3' not found"})

    client = OllamaClient("http://ollama", "llama3", 5, transport=httpx.MockTransport(missing))
    with pytest.raises(LLMError, match="ollama pull llama3"):
        client.complete("x")


def test_cost_uses_configured_prices():
    from app.config import settings

    result = FakeLLMClient().complete("x" * 4000)
    expected = (1000 * settings.llm_price_input_per_mtok + result.output_tokens * settings.llm_price_output_per_mtok) / 1e6
    assert result.cost_usd == pytest.approx(expected)


def test_fake_returns_valid_classification_json_and_cited_draft():
    fake = FakeLLMClient()
    assert extract_json(fake.complete("x", json_mode=True).text)["team"] == "Product"
    assert "[S1]" in fake.complete("x").text


def test_extract_json_handles_fences_and_garbage():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("Sure! {\"a\": 2} hope that helps") == {"a": 2}
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None
