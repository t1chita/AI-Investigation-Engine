"""LLM clients behind one interface: ``complete_json(system, user) -> dict``.

Every call asks the model for a single JSON object. The investigation pipeline validates that
object itself (see ``investigate.py``), so providers only need a "JSON mode", not native tool
calling, and can be swapped by configuration.

Errors are normalised to ``LLMError`` subclasses. Transient ones (rate limit, timeout, 5xx) are
retried with exponential backoff; the caller decides what to do when retries run out.
Token usage is counted per client so the UI and the evaluation can report LLM cost.
"""

import json
import time
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    """Any failure talking to the model."""


class LLMRateLimited(LLMError):
    pass


class LLMQuotaExhausted(LLMRateLimited):
    """A daily quota (tokens or requests per day) is used up. Retrying minutes later cannot help."""


class LLMUnavailable(LLMError):
    """Timeout, network error or 5xx."""


class LLMBadOutput(LLMError):
    """The model answered, but not with a usable JSON object."""


class LLMAuthError(LLMError):
    pass


class LLMBillingError(LLMError):
    """The provider refuses to serve the account (no credits / billing problem). Not retried."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, tolerating Markdown code fences around it."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        raise LLMBadOutput("no JSON object in model output")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMBadOutput(f"invalid JSON from model: {exc.msg}") from None
    if not isinstance(value, dict):
        raise LLMBadOutput("model output is not a JSON object")
    return value


class _HTTPClient:
    """Shared retry, timeout and error mapping for HTTP providers."""

    provider = "http"

    def __init__(self, api_key: str, model: str, timeout: float = 45.0, max_retries: int = 3,
                 max_wait: float = 120.0, http=None):
        if not api_key:
            raise LLMAuthError(f"{self.provider}: LLM_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.max_wait = max_wait  # total seconds one call may spend waiting between retries
        self.http = http or httpx.Client(timeout=timeout)
        self.usage = Usage()

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        delay = 2.0
        deadline = time.monotonic() + self.max_wait
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                error: LLMError = LLMUnavailable(f"{self.provider}: {type(exc).__name__}")
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (401, 403):
                    raise LLMAuthError(f"{self.provider}: authentication failed ({response.status_code})")
                if response.status_code == 402:
                    raise LLMBillingError(f"{self.provider}: account has no usable credits or billing (402)")
                if response.status_code == 429:
                    if "per day" in response.text.lower():  # e.g. Groq "tokens per day (TPD)"
                        raise LLMQuotaExhausted(f"{self.provider}: daily quota used up; try again tomorrow")
                    error = LLMRateLimited(f"{self.provider}: rate limited")
                    retry_after = response.headers.get("retry-after")
                    if retry_after and retry_after.replace(".", "", 1).isdigit():
                        delay = max(delay, min(float(retry_after) + 0.5, 60.0))
                elif response.status_code >= 500:
                    error = LLMUnavailable(f"{self.provider}: server error {response.status_code}")
                else:
                    raise LLMError(f"{self.provider}: request rejected ({response.status_code}): {response.text[:200]}")
            if attempt == self.max_retries or time.monotonic() + delay > deadline:
                raise error
            self.usage.retries += 1
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise AssertionError("unreachable")


class GroqClient(_HTTPClient):
    """Groq's OpenAI-compatible chat completions API with JSON mode."""

    provider = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    DEFAULT_FALLBACKS = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")

    def __init__(self, api_key: str, model: str = "", fallback_models: list[str] | None = None, **kw):
        # Free-tier key (October 2026), per model: 1,000 requests/day, 8,000 tokens/minute and
        # 200,000 tokens/day. Each model has its OWN daily quota, so when the main model's daily
        # quota is used up we continue with the next model in the list (and say so).
        kw.setdefault("max_retries", 6)  # the per-minute token window resets within a minute
        super().__init__(api_key, model or "openai/gpt-oss-120b", **kw)
        fallbacks = self.DEFAULT_FALLBACKS if fallback_models is None else fallback_models
        self.models = [self.model] + [m for m in fallbacks if m and m != self.model]

    def complete_json(self, system: str, user: str) -> dict:
        while True:
            try:
                return self._complete(system, user)
            except LLMQuotaExhausted:
                position = self.models.index(self.model)
                if position + 1 >= len(self.models):
                    raise
                self.model = self.models[position + 1]  # sticky: later calls use it too

    def _complete(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data = self._post(self.url, {"Authorization": f"Bearer {self.api_key}"}, body)
        self.usage.calls += 1
        usage = data.get("usage") or {}
        self.usage.input_tokens += usage.get("prompt_tokens", 0)
        self.usage.output_tokens += usage.get("completion_tokens", 0)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMBadOutput("groq: unexpected response shape") from None
        return parse_json_object(text)


class GeminiClient(_HTTPClient):
    """Google Gemini generateContent API with a JSON response MIME type."""

    provider = "gemini"

    def __init__(self, api_key: str, model: str = "", **kw):
        # gemini-2.5-flash is closed to new API keys (404, October 2026); Google points to 3.8-flash.
        super().__init__(api_key, model or "gemini-3.8-flash", **kw)

    def complete_json(self, system: str, user: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        data = self._post(url, {"x-goog-api-key": self.api_key}, body)
        self.usage.calls += 1
        meta = data.get("usageMetadata") or {}
        self.usage.input_tokens += meta.get("promptTokenCount", 0)
        self.usage.output_tokens += meta.get("candidatesTokenCount", 0)
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise LLMBadOutput("gemini: no content in response (possibly blocked)") from None
        return parse_json_object(text)


class ScriptedLLM:
    """Test double: returns queued responses (dicts, or exceptions to raise) and records prompts."""

    provider = "scripted"
    model = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []
        self.usage = Usage()

    def complete_json(self, system: str, user: str) -> dict:
        self.prompts.append((system, user))
        self.usage.calls += 1
        self.usage.input_tokens += (len(system) + len(user)) // 4  # rough 4-chars-per-token estimate
        self.usage.output_tokens += 50
        if not self.responses:
            raise LLMBadOutput("scripted LLM has no response left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(provider: str, api_key: str = "", model: str = "", fallback_models: list[str] | None = None):
    """Build the configured client. ``none``/``fake`` means no LLM (retrieval-only answers)."""
    provider = (provider or "").lower()
    if provider == "groq":
        return GroqClient(api_key, model, fallback_models)
    if provider == "gemini":
        return GeminiClient(api_key, model)
    if provider in ("", "none", "fake"):
        return None
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}; use groq, gemini or none")
