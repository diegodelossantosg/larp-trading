"""End-to-end run of trader.main.trade with fake Alpaca and Claude."""

import pandas as pd

from trader import journal, main
from trader.brain import Decision, Usage
from trader.config import Config


def cfg(**kw):
    base = dict(anthropic_api_key="", claude_backend="api", alpaca_api_key="k", alpaca_secret_key="s", paper=True,
                claude_model="claude-opus-5-5", claude_effort="high", idle_asset="SPY",
                max_position_weight=0.30, max_positions=8, max_leveraged_weight=0.10,
                max_daily_turnover=1.0, rebalance_band=0.02, min_order_usd=25,
                max_drawdown_halt=0.25, trading_enabled=True, max_web_searches=5, capital_usd=0,
                max_idle_weight=1.0, stop_trail_pct=0.06, watch_min_score=5, watch_buy_size=0.08,
                watch_max_buys_per_day=4, watch_require_claude=True, min_hold_days=3)
    base.update(kw)
    return Config(**base)


def frame(n=250):
    close = pd.Series([100 + i * 0.1 for i in range(n)])
    return pd.DataFrame({"open": close - 0.05, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1_000_000})


class FakeMarket:
    def daily_bars(self, symbols, days=400):
        return {s: frame() for s in symbols if s != "NOPE"}

    def news(self, **kw):
        return [{"headline": "Fed holds rates"}]


class FakeBroker:
    def __init__(self):
        self.sent = []

    def market_open(self):
        return True

    def account(self):
        return {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}

    def positions(self):
        return {}

    def vet(self, s):
        if s == "NOPE":
            return False, False, "", "unknown symbol"
        return True, False, ("small" if s == "TINY" else "etf"), ""

    def execute(self, orders, prices, positions):
        self.sent = orders
        return [{"symbol": o.symbol, "side": o.side, "notional": o.notional, "status": "accepted"} for o in orders]


def fake_decide(data):
    return lambda cfg, briefing: Decision(data, Usage(model="claude-opus-5-5", input_tokens=1000, web_searches=2))


def test_trade_places_orders_and_journals(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "decide", fake_decide({
        "market_view": "Energy strong.", "key_drivers": [], "trade_today": True, "confidence": "medium",
        "watchlist": [], "positions": [
            {"symbol": "XLE", "weight": 0.5, "thesis": "oil", "signals": [], "exit_plan": "x"},
            {"symbol": "NOPE", "weight": 0.1, "thesis": "?", "signals": [], "exit_plan": "x"},
        ]}))
    broker = FakeBroker()
    assert main.trade(cfg(), broker, FakeMarket(), dry_run=False, force=False) == 0

    rec = journal.read("decisions.json", [])[-1]
    assert rec["final_targets"] == {"XLE": 0.3, "SPY": 0.7}
    assert {o.symbol for o in broker.sent} == {"XLE", "SPY"}
    assert any("NOPE" in n for n in rec["notes"])
    assert rec["ai_cost_usd"] > 0
    assert journal.read("status.json", {})["mode"] == "paper"


def test_dry_run_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "decide", fake_decide({
        "market_view": "-", "key_drivers": [], "trade_today": True, "confidence": "low",
        "watchlist": [], "positions": []}))
    broker = FakeBroker()
    main.trade(cfg(), broker, FakeMarket(), dry_run=True, force=False)
    rec = journal.read("decisions.json", [])[-1]
    assert broker.sent == [] and rec["orders"][0]["status"] == "not sent"


def test_drawdown_halt(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    journal.write("equity.json", [{"date": "2026-01-01", "equity": 20_000, "cash": 0, "spy_close": 500}])
    monkeypatch.setattr(main, "decide", lambda *a: (_ for _ in ()).throw(AssertionError("called")))
    assert main.trade(cfg(), FakeBroker(), FakeMarket(), dry_run=False, force=False) == 1
    assert "HALTED" in journal.read("decisions.json", [])[-1]["error"]


def test_capital_limits_agent_to_its_own_money(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "decide", fake_decide({
        "market_view": "-", "key_drivers": [], "trade_today": True, "confidence": "low", "watchlist": [],
        "positions": [{"symbol": "XLE", "weight": 0.3, "thesis": "", "signals": [], "exit_plan": ""}]}))
    broker = FakeBroker()  # the Alpaca account itself holds $10,000
    main.trade(cfg(capital_usd=1000), broker, FakeMarket(), dry_run=False, force=False)
    total = sum(o.notional for o in broker.sent)
    assert 950 < total <= 1000
    assert journal.read("capital.json", {})["baseline_equity"] == 10_000

    # Later the account gained $50: the agent now has $1,050.
    broker.account = lambda: {"equity": 10_050.0, "cash": 0.0, "buying_power": 0.0}
    assert main._account(cfg(capital_usd=1000), broker, {})["equity"] == 1050


def test_hold_still_moves_spare_money_to_new_idle_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "decide", fake_decide({
        "market_view": "-", "key_drivers": [], "trade_today": False, "confidence": "low", "watchlist": [],
        "positions": [{"symbol": "XLE", "weight": 0.5, "thesis": "", "signals": [], "exit_plan": ""}]}))
    broker = FakeBroker()
    broker.positions = lambda: {
        "SPY": {"qty": 1, "market_value": 5000.0, "avg_entry_price": 1, "current_price": 1, "unrealized_pl": 0, "unrealized_plpc": 0},
        "XLE": {"qty": 1, "market_value": 4900.0, "avg_entry_price": 1, "current_price": 1, "unrealized_pl": 0, "unrealized_plpc": 0}}
    main.trade(cfg(idle_asset="SSO", max_position_weight=0.5, max_idle_weight=0.5), broker, FakeMarket(),
               dry_run=False, force=False)
    sent = {(o.side, o.symbol) for o in broker.sent}
    assert ("sell", "SPY") in sent and ("buy", "SSO") in sent
    assert not any(o.symbol == "XLE" for o in broker.sent)  # XLE is already at target


def test_idle_asset_switches_to_plain_spy_below_the_200_day_average(tmp_path, monkeypatch):
    import pandas as pd
    from trader import journal, main
    up = pd.Series(range(100, 400), dtype=float)
    assert main.idle_for(up, "SSO", "SPY")[0] == "SSO"
    down = pd.Series(list(range(100, 400)) + [200.0] * 5)
    assert main.idle_for(down, "SSO", "SPY")[0] == "SPY"
    assert main.idle_for(up.iloc[:50], "SSO", "SPY")[0] == "SSO"          # too little history: usual one

    monkeypatch.setattr("trader.journal.DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_today", lambda: "2026-10-23")
    idx = pd.date_range("2025-09-01", periods=305, freq="D", tz="UTC") + pd.Timedelta(hours=4)
    bars = pd.DataFrame({"close": list(range(100, 400)) + [200.0] * 5}, index=idx)
    calls = []

    class FakeMarket:
        def daily_bars(self, symbols, days=400):
            calls.append(symbols)
            return {"SPY": bars}

    c = main._with_todays_idle(cfg(), FakeMarket())
    assert c.idle_asset == "SPY" and journal.read("idle.json", {})["asset"] == "SPY"
    assert main._with_todays_idle(cfg(), FakeMarket()).idle_asset == "SPY" and len(calls) == 1   # once a day
