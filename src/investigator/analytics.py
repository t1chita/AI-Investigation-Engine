"""Aggregate recorded LLM usage into the numbers the Analytics page shows.

Pure functions over the rows from ``Registry.usage_rows`` so they are easy to test.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from .pricing import EVAL_RUN1, FREE_TIER, PRICES, PRICES_CHECKED, cost, price_for

MIN_LIVE_QUESTIONS = 5  # fewer live questions than this: use the evaluation's averages as the baseline


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)  # SQLite drops the time zone


def summarize(rows: list[dict], now: datetime | None = None, days: int = 14) -> dict:
    now = now or datetime.now(timezone.utc)
    questions = len(rows)
    calls = sum(r["calls"] for r in rows)
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)

    by_model: dict[str, dict] = defaultdict(lambda: {"questions": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0})
    for r in rows:
        m = by_model[r["model"]]
        m["questions"] += 1
        m["calls"] += r["calls"]
        m["input_tokens"] += r["input_tokens"]
        m["output_tokens"] += r["output_tokens"]
    models = []
    for name, m in sorted(by_model.items(), key=lambda kv: -kv[1]["questions"]):
        price = price_for(name)
        models.append({"model": name, **m, "list_cost": cost(m["input_tokens"], m["output_tokens"], price) if price else None})

    # Daily totals for the last `days` days (UTC), including empty days so the chart has no gaps.
    first = (now - timedelta(days=days - 1)).date()
    daily = {first + timedelta(days=i): {"questions": 0, "input_tokens": 0, "output_tokens": 0} for i in range(days)}
    for r in rows:
        day = _aware(r["created_at"]).date()
        if day in daily:
            d = daily[day]
            d["questions"] += 1
            d["input_tokens"] += r["input_tokens"]
            d["output_tokens"] += r["output_tokens"]

    # Groq's daily limit is a rolling 24 hours per model.
    since = now - timedelta(hours=24)
    last_24h = Counter()
    for r in rows:
        if _aware(r["created_at"]) >= since:
            last_24h[r["model"]] += r["input_tokens"] + r["output_tokens"]

    if questions >= MIN_LIVE_QUESTIONS:
        basis = {"source": f"live app, {questions} questions", "input_tokens": tin / questions,
                 "output_tokens": tout / questions, "calls": calls / questions}
    else:
        n = EVAL_RUN1["questions"]
        basis = {"source": f"evaluation run 1, {n} questions", "input_tokens": EVAL_RUN1["input_tokens"] / n,
                 "output_tokens": EVAL_RUN1["output_tokens"] / n, "calls": EVAL_RUN1["calls"] / n}

    prices = [
        {**p, "cost_per_question": cost(basis["input_tokens"], basis["output_tokens"], p)}
        for p in PRICES
    ]
    return {
        "totals": {
            "questions": questions, "calls": calls, "input_tokens": tin, "output_tokens": tout,
            "avg_seconds": round(sum(r["seconds"] for r in rows) / questions, 1) if questions else None,
            "paid": 0.0,  # free tier
            "list_cost": sum(m["list_cost"] or 0 for m in models),
        },
        "by_model": models,
        "by_status": dict(Counter(r["status"] for r in rows)),
        "by_mode": dict(Counter(r["mode"] for r in rows)),
        "daily": [{"day": d.isoformat(), **v} for d, v in daily.items()],
        "last_24h": {"tokens_by_model": dict(last_24h), "limit_per_model": FREE_TIER["tokens_per_day"]},
        "free_tier": FREE_TIER,
        "basis": {k: round(v, 1) if isinstance(v, float) else v for k, v in basis.items()},
        "prices": prices,
        "prices_checked": PRICES_CHECKED,
        "evaluation": EVAL_RUN1,
    }
