"""The Claude loop: resumes pause_turn, nudges when no decision, handles refusals."""

from types import SimpleNamespace as NS

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
    monkeypatch.setattr(brain.anthropic, "Anthropic", fake_client([
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
    monkeypatch.setattr(brain.anthropic, "Anthropic", fake_client(
        [NS(stop_reason="refusal", content=[], usage=usage())], []))
    d = brain.decide(cfg(), {})
    assert d.data is None and "declined" in d.error
