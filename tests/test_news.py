"""News Hunter: Opus's ideas pass through the same safety rules before anything is bought."""

from trader import journal, main
from trader.watch import Quote

from tests.test_trade_flow import FakeBroker, FakeMarket, cfg


class NewsMarket(FakeMarket):
    def news(self, **kw):
        return [{"time": "2026-10-15 14:00 UTC", "headline": "Acme wins $5B Pentagon contract",
                 "summary": "", "symbols": ["ACME", "NVDA"], "source": "benzinga"}]

    def quotes(self, symbols):
        return {s: Quote(100.0, 98.0, 0, 1) for s in symbols}


class HeldBroker(FakeBroker):
    def positions(self):
        return {"SSO": {"qty": 1, "market_value": 5000.0, "avg_entry_price": 1, "current_price": 1,
                        "unrealized_plpc": 0.0}}

    def account(self):
        return {"equity": 10_000.0, "cash": 0.0, "buying_power": 0.0}

    def vet(self, s):
        if s == "TINY":
            return True, False, "small", ""
        return super().vet(s) if s not in ("NVDA", "AAPL") else (True, False, "big", "")


def run(tmp_path, monkeypatch, ideas):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_today", lambda: "2026-10-15")
    monkeypatch.setattr(main, "claude_available", lambda cfg: True)
    monkeypatch.setattr(main, "hunt_news", lambda cfg, ctx: ({"summary": "One real opportunity.", "ideas": ideas}, None))
    broker = HeldBroker()
    assert main.news(cfg(idle_asset="SSO"), broker, NewsMarket(), dry_run=False, force=True) == 0
    return broker, journal.read("decisions.json", [])[-1]


def idea(sym, earnings="2026-12-01"):
    return {"symbol": sym, "headline": "x", "why_not_priced_in": "x", "plain_reason": "Won a huge contract.",
            "next_earnings": earnings}


def test_news_buys_are_sized_funded_and_journaled(tmp_path, monkeypatch):
    broker, rec = run(tmp_path, monkeypatch, [idea("NVDA"), idea("TINY")])
    sent = {(o.symbol, o.side): o.notional for o in broker.sent}
    assert sent[("NVDA", "buy")] == 500 and sent[("TINY", "buy")] == 500   # 5% each
    assert sent[("SSO", "sell")] > 0                                        # paid for by selling SSO
    assert rec["source"] == "news" and rec["plain_summary"] == "One real opportunity."
    assert journal.read("news.json", {})["buys"]["2026-10-15"] == 2


def test_news_respects_earnings_and_daily_limit(tmp_path, monkeypatch):
    broker, rec = run(tmp_path, monkeypatch, [idea("AAPL", "2026-10-16"), idea("NVDA"), idea("TINY")])
    bought = [o.symbol for o in broker.sent if o.side == "buy"]
    assert bought == ["NVDA"]                                    # AAPL reports tomorrow; only 2 ideas looked at
    assert any("AAPL" in n and "earnings" in n for n in rec["notes"])


def test_earnings_beat_with_unknown_next_date_can_buy(tmp_path, monkeypatch):
    beat = dict(idea("NVDA", "unknown"), idea_type="earnings_beat")
    drop = dict(idea("AAPL", "unknown"), idea_type="overreaction")
    broker, rec = run(tmp_path, monkeypatch, [beat, drop])
    assert [o.symbol for o in broker.sent if o.side == "buy"] == ["NVDA"]   # just reported: next one is months away
    assert any("AAPL" in n and "isn't confirmed" in n for n in rec["notes"])  # other ideas still need a date


def test_earnings_beats_are_kept_longer(tmp_path, monkeypatch):
    beat = dict(idea("NVDA"), idea_type="earnings_beat")
    drop = dict(idea("TINY"), idea_type="overreaction")
    broker, rec = run(tmp_path, monkeypatch, [beat, drop])
    held = {o["symbol"]: o["hold_days"] for o in rec["orders"] if o["side"] == "buy"}
    assert held == {"NVDA": 20, "TINY": 10}
    monkeypatch.setattr(main, "_today", lambda: "2026-11-02")    # 12 trading days later
    assert main._recent_watch_buys(3, 10) == {"NVDA"}
