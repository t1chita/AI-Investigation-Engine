"""LLM prices and cost arithmetic for the Analytics page.

cost = input_tokens / 1,000,000 x input price + output_tokens / 1,000,000 x output price

Prices are US dollars per 1 million tokens, standard (non-batch, uncached) paid rates. They change
often: they were collected on PRICES_CHECKED from the listed sources (mostly third-party price
lists, which disagreed in places) and must be checked on the provider's own page before any
budget decision. Our live demo runs on Groq's free tier, so what it actually pays is 0.

Only the Groq models were run on our evaluation. For the others, the page shows what the SAME
token counts would cost; their answer quality on our questions is not measured.
"""

PRICES_CHECKED = "2026-10-29"

PRICES = [
    # provider, model, input $/1M, output $/1M, note
    {"provider": "Groq", "model": "openai/gpt-oss-120b", "input": 0.15, "output": 0.60,
     "note": "our main model (free tier in use)", "source": "https://groq.com/pricing"},
    {"provider": "Groq", "model": "openai/gpt-oss-20b", "input": 0.075, "output": 0.30,
     "note": "our first fallback", "source": "https://groq.com/pricing"},
    {"provider": "Groq", "model": "qwen/qwen3.8-27b", "input": 0.80, "output": 4.00,
     "note": "our second fallback (preview)", "source": "https://groq.com/pricing"},
    {"provider": "OpenAI", "model": "gpt-5-nano", "input": 0.05, "output": 0.40,
     "note": "", "source": "https://openai.com/api/pricing"},
    {"provider": "OpenAI", "model": "gpt-5.4-mini", "input": 0.75, "output": 4.50,
     "note": "", "source": "https://openai.com/api/pricing"},
    {"provider": "Google", "model": "gemini-2.5-flash-lite", "input": 0.10, "output": 0.40,
     "note": "", "source": "https://ai.google.dev/gemini-api/docs/pricing"},
    {"provider": "Google", "model": "gemini-2.5-flash", "input": 0.30, "output": 2.50,
     "note": "", "source": "https://ai.google.dev/gemini-api/docs/pricing"},
    {"provider": "Anthropic", "model": "claude-haiku-4.5", "input": 1.00, "output": 5.00,
     "note": "", "source": "https://docs.claude.com/en/docs/about-claude/pricing"},
    {"provider": "Anthropic", "model": "claude-sonnet-4.6", "input": 3.00, "output": 15.00,
     "note": "", "source": "https://docs.claude.com/en/docs/about-claude/pricing"},
]

# Groq free tier, per model (October 2026). The live app and the evaluation share one key.
FREE_TIER = {"requests_per_day": 1000, "tokens_per_minute": 8000, "tokens_per_day": 200_000}

# Measured on evaluation run 1 (46 questions x 2 configurations = 92 questions, gpt-oss-120b).
# Used as the per-question baseline until the live app has its own history.
EVAL_RUN1 = {"questions": 92, "calls": 185, "input_tokens": 142_396, "output_tokens": 40_416}


def cost(input_tokens: float, output_tokens: float, price: dict) -> float:
    """Dollar cost of the given token counts at one model's price."""
    return input_tokens / 1e6 * price["input"] + output_tokens / 1e6 * price["output"]


def price_for(model: str) -> dict | None:
    return next((p for p in PRICES if p["model"] == model), None)
