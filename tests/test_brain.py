"""The Claude loop: resumes pause_turn, nudges when no decision, handles refusals."""

from types import SimpleNamespace as NS

import anthropic

from trader import brain
from tests.test_trade_flow import cfg


def usage(searches=0):
    return NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
              cache_creation_input_tokens=0, server_tool_use=NS(web_search_requests=searches))


def fake_client(responses, calls):
    class Stream:
        def __init__(self, resp):
            self.resp = resp

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return self.resp

    def stream(**kwargs):
        calls.append(kwargs)
        return Stream(responses.pop(0))

    return lambda **kw: NS(beta=NS(messages=NS(stream=stream)))


def test_resumes_pause_turn_then_returns_decision(monkeypatch):
    search = NS(type="web_search_tool_result", content=[NS(type="web_search_result", title="Fed", url="https://x")])
    decision = NS(type="tool_use", name="submit_decision", input={"trade_today": False})
    calls = []
    monkeypatch.setattr(anthropic, "Anthropic", fake_client([
        NS(stop_reason="pause_turn", content=[search], usage=usage(3)),
        NS(stop_reason="tool_use", content=[decision], usage=usage()),
    ], calls))
    d = brain.decide(cfg(), {"date": "2026-10-12"})
    assert d.data == {"trade_today": False}
    assert d.usage.web_searches == 3 and d.usage.cost_usd() > 0
    assert d.sources == [{"title": "Fed", "url": "https://x"}]
    assert len(calls) == 2 and calls[1]["messages"][-1]["role"] == "assistant"
    assert calls[0]["fallbacks"] == "default"


def test_refusal_returns_error(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", fake_client(
        [NS(stop_reason="refusal", content=[], usage=usage())], []))
    d = brain.decide(cfg(), {})
    assert d.data is None and "declined" in d.error


def test_subscription_backend_parses_claude_code_output(monkeypatch):
    import json
    seen = {}

    def fake_run(cmd, input, **kw):
        seen["cmd"], seen["input"] = cmd, input
        out = {"is_error": False, "subtype": "success", "total_cost_usd": 0.42,
               "usage": {"input_tokens": 10, "output_tokens": 5},
               "structured_output": {"trade_today": False, "sources": [{"title": "Fed", "url": "https://x"}]}}
        return NS(stdout=json.dumps(out), stderr="", returncode=0)

    monkeypatch.setattr(brain.subprocess, "run", fake_run)
    d = brain.decide(cfg(claude_backend="subscription", claude_model="sonnet"), {"date": "2026-10-12"})
    assert d.data == {"trade_today": False}
    assert d.sources == [{"title": "Fed", "url": "https://x"}]
    assert d.usage.cost_usd() == 0.0 and d.usage.api_equivalent_usd == 0.42
    assert "--json-schema" in seen["cmd"] and "2026-10-12" in seen["input"]


def test_subscription_backend_reports_failures(monkeypatch):
    monkeypatch.setattr(brain.subprocess, "run",
                        lambda *a, **k: NS(stdout="", stderr="Invalid API key · Please run /login", returncode=1))
    d = brain.decide(cfg(claude_backend="subscription"), {})
    assert d.data is None and "Invalid API key" in d.error
