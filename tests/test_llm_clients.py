"""HTTP clients against a fake transport: retries, error mapping, usage counting. No network."""

import httpx
import pytest

from investigator import llm as llm_mod
from investigator.llm import (
    GeminiClient, GroqClient, LLMAuthError, LLMBadOutput, LLMBillingError, LLMQuotaExhausted, LLMRateLimited,
    make_llm,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def client_with(cls, responses):
    calls = []

    def handler(request):
        calls.append(request)
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    return cls("test-key", http=httpx.Client(transport=httpx.MockTransport(handler))), calls


GROQ_OK = {"choices": [{"message": {"content": '{"queries": ["a"]}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def test_groq_retries_rate_limit_then_succeeds():
    client, calls = client_with(GroqClient, [(429, {}), (503, {}), (200, GROQ_OK)])
    assert client.complete_json("s", "u") == {"queries": ["a"]}
    assert len(calls) == 3 and client.usage.retries == 2
    assert (client.usage.input_tokens, client.usage.output_tokens) == (10, 3)
    assert calls[0].headers["authorization"] == "Bearer test-key"


def test_groq_gives_up_after_max_retries():
    client, calls = client_with(GroqClient, [(429, {})] * 7)
    with pytest.raises(LLMRateLimited):
        client.complete_json("s", "u")
    assert len(calls) == 7  # first try + 6 retries


def test_daily_quota_is_not_retried():
    daily = {"error": {"message": "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199535"}}
    client, calls = client_with(GroqClient, [(429, daily)])
    with pytest.raises(LLMQuotaExhausted, match="daily quota"):
        client.complete_json("s", "u")
    assert len(calls) == 1


def test_retries_stop_at_the_wait_budget(monkeypatch):
    clock = iter(range(0, 10_000, 50))  # every monotonic() call advances 50 s
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: next(clock))
    client, calls = client_with(GroqClient, [(429, {})] * 7)
    client.max_wait = 120
    with pytest.raises(LLMRateLimited):
        client.complete_json("s", "u")
    assert len(calls) < 7  # gave up before using all 6 retries


def test_auth_and_billing_errors_are_not_retried():
    client, calls = client_with(GroqClient, [(401, {})])
    with pytest.raises(LLMAuthError):
        client.complete_json("s", "u")
    assert len(calls) == 1
    client, calls = client_with(GeminiClient, [(402, {})])
    with pytest.raises(LLMBillingError):
        client.complete_json("s", "u")
    assert len(calls) == 1


def test_gemini_parses_parts_and_reports_blocked_output():
    ok = {"candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}], "usageMetadata": {"promptTokenCount": 7}}
    client, calls = client_with(GeminiClient, [(200, ok), (200, {"candidates": []})])
    assert client.complete_json("s", "u") == {"x": 1}
    assert calls[0].headers["x-goog-api-key"] == "test-key"
    with pytest.raises(LLMBadOutput):
        client.complete_json("s", "u")


def test_make_llm_needs_a_key_and_known_provider():
    assert make_llm("none") is None
    with pytest.raises(LLMAuthError):
        make_llm("groq", api_key="")
    with pytest.raises(ValueError):
        make_llm("openai-ish", api_key="k")
