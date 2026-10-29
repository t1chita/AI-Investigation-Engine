"""Cost arithmetic and usage aggregation for the Analytics page (hand-checkable numbers)."""

from datetime import datetime, timedelta, timezone

import pytest

from investigator.analytics import summarize
from investigator.pricing import EVAL_RUN1, PRICES, cost, price_for
from investigator.registry import Registry

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def test_cost_formula():
    price = {"input": 0.15, "output": 0.60}
    # 1,000,000 input tokens = $0.15; 500,000 output tokens = $0.30
    assert cost(1_000_000, 500_000, price) == pytest.approx(0.45)
    assert cost(0, 0, price) == 0
    assert price_for("openai/gpt-oss-120b")["output"] == 0.60
    assert price_for("no-such-model") is None
    assert all(p["input"] > 0 and p["output"] > 0 for p in PRICES)


def row(model, tin, tout, at, status="answered", calls=2):
    return {"model": model, "status": status, "mode": "hybrid", "calls": calls,
            "input_tokens": tin, "output_tokens": tout, "seconds": 4.0, "created_at": at}


def test_summary_totals_days_and_rolling_window():
    rows = [
        row("openai/gpt-oss-120b", 1000, 200, NOW - timedelta(days=3)),
        row("openai/gpt-oss-120b", 3000, 400, NOW - timedelta(hours=2)),
        row("openai/gpt-oss-20b", 500, 100, NOW - timedelta(hours=1), status="conflict"),
    ]
    s = summarize(rows, now=NOW, days=7)
    assert s["totals"]["questions"] == 3 and s["totals"]["input_tokens"] == 4500 and s["totals"]["output_tokens"] == 700
    assert s["totals"]["paid"] == 0
    # list cost: 120b 4000 in / 600 out -> 0.0006 + 0.00036; 20b 500 / 100 -> 0.0000375 + 0.00003
    assert s["totals"]["list_cost"] == pytest.approx(0.0010275)
    assert [m["model"] for m in s["by_model"]] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert s["by_status"] == {"answered": 2, "conflict": 1}
    assert len(s["daily"]) == 7 and s["daily"][-1]["questions"] == 2 and s["daily"][-4]["questions"] == 1
    # only the last 24 hours count against the daily limit
    assert s["last_24h"]["tokens_by_model"] == {"openai/gpt-oss-120b": 3400, "openai/gpt-oss-20b": 600}


def test_baseline_switches_from_evaluation_to_live_data():
    few = summarize([row("m", 100, 10, NOW)], now=NOW)
    assert few["basis"]["source"].startswith("evaluation")
    assert few["basis"]["input_tokens"] == round(EVAL_RUN1["input_tokens"] / EVAL_RUN1["questions"], 1)
    many = summarize([row("m", 100, 10, NOW)] * 5, now=NOW)
    assert many["basis"]["source"].startswith("live") and many["basis"]["input_tokens"] == 100
    p = next(p for p in many["prices"] if p["model"] == "openai/gpt-oss-120b")
    assert p["cost_per_question"] == pytest.approx(cost(100, 10, p))


def test_registry_stores_usage(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    reg.record_usage("openai/gpt-oss-120b", "answered", "hybrid", 3, 2500, 600, 7.5, at=NOW)
    rows = reg.usage_rows()
    assert len(rows) == 1 and rows[0]["input_tokens"] == 2500 and rows[0]["calls"] == 3
    assert summarize(rows, now=NOW)["daily"][-1]["output_tokens"] == 600
