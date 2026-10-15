"""Company stocks: size caps, the earnings rule, and the watcher's stock handling."""

from trader import journal, main
from trader.risk import apply_limits, earnings_blackout
from trader.watch import Quote, buy_candidates, stop_orders, update_highs

from tests.test_trade_flow import FakeBroker, FakeMarket, cfg, fake_decide


def limits(proposed, stocks):
    return apply_limits(proposed, allowed=set(proposed), leveraged=set(), idle_asset="SSO",
                        max_position_weight=0.30, max_positions=12, max_leveraged_weight=0.10,
                        max_idle_weight=0.50, stocks=stocks)


def test_stock_caps_per_company_small_total_and_all_stocks():
    targets, notes = limits({"NVDA": 0.20, "TINY": 0.08}, {"NVDA": "big", "TINY": "small"})
    assert targets["NVDA"] == 0.10 and targets["TINY"] == 0.05

    small = {f"S{i}": "small" for i in range(4)}
    targets, _ = limits({s: 0.05 for s in small}, small)
    assert abs(sum(targets[s] for s in small) - 0.10) < 1e-3

    big = {s: "big" for s in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META")}
    targets, notes = limits({s: 0.10 for s in big} | {"XLE": 0.2}, big)
    assert abs(sum(targets[s] for s in big) - 0.40) < 1e-3
    assert targets["SSO"] <= 0.50 and any("single-company stocks" in n for n in notes)


def test_earnings_rule_blocks_adding_but_allows_holding():
    proposed = {"NVDA": 0.10, "AAPL": 0.10, "MSFT": 0.10, "XLE": 0.2}
    current = {"NVDA": 0.04}
    earnings = {"NVDA": "2026-10-19", "AAPL": "2026-11-20", "MSFT": "unknown", "XLE": "n/a"}
    out, notes = earnings_blackout(proposed, current, {"NVDA", "AAPL", "MSFT"}, earnings, "2026-10-15", 3)
    assert out["NVDA"] == 0.04          # reports in 2 trading days: keep what we have, no more
    assert out["AAPL"] == 0.10          # far away: fine
    assert out["MSFT"] == 0.0           # unknown date: don't buy
    assert out["XLE"] == 0.2 and len(notes) == 2


def test_trade_buys_stocks_within_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_today", lambda: "2026-10-15")
    monkeypatch.setattr(main, "decide", fake_decide({
        "market_view": "-", "key_drivers": [], "trade_today": True, "confidence": "medium", "watchlist": [],
        "positions": [
            {"symbol": "NVDA", "weight": 0.15, "thesis": "", "signals": [], "exit_plan": "", "next_earnings": "2026-11-19"},
            {"symbol": "TINY", "weight": 0.08, "thesis": "", "signals": [], "exit_plan": "", "next_earnings": "2026-12-01"},
            {"symbol": "MCD", "weight": 0.08, "thesis": "", "signals": [], "exit_plan": "", "next_earnings": "2026-10-16"},
            {"symbol": "XLE", "weight": 0.30, "thesis": "", "signals": [], "exit_plan": "", "next_earnings": "n/a"},
        ]}))
    main.trade(cfg(idle_asset="SSO", max_idle_weight=0.5), FakeBroker(), FakeMarket(), dry_run=False, force=False)
    rec = journal.read("decisions.json", [])[-1]
    t = rec["final_targets"]
    assert t["NVDA"] == 0.10 and t["TINY"] == 0.05 and "MCD" not in t
    assert any("MCD" in n and "earnings" in n for n in rec["notes"])


def test_stocks_get_a_wider_stop_than_etfs():
    positions = {s: {"market_value": 92, "avg_entry_price": 100, "current_price": 92} for s in ("NVDA", "XLE")}
    quotes = {"NVDA": Quote(92.0, 93, 0, 1), "XLE": Quote(92.0, 93, 0, 1)}
    highs = update_highs({}, positions, quotes)
    orders = stop_orders(positions, highs, quotes, "SSO", 0.06, stocks={"NVDA"}, stock_trail_pct=0.10)
    assert [o.symbol for o in orders] == ["XLE"]  # 8% down: ETF stopped, stock not yet


def test_watcher_respects_stock_caps():
    strong = dict(price=103.0, prev_close=100.0, high_20d=102.0, atr_pct=1.5, sma20=98.0, sma50=95.0,
                  vwap=101.5, candles=["bullish_engulfing"], volume_ratio=2.0, bars_today=6)
    quotes = {"SPY": Quote(500.5, 500, 510, 1, bars_today=6), "NVDA": Quote(**strong), "AAPL": Quote(**strong),
              "MSFT": Quote(**strong)}
    found = buy_candidates(quotes, {"NVDA": 0.05}, "SSO", 5, 0.30, 0.08)
    assert [s for s, _, _ in found] == ["AAPL", "MSFT"]  # NVDA would pass 10%
    found = buy_candidates(quotes, {"GOOGL": 0.10, "AMZN": 0.10, "META": 0.10}, "SSO", 5, 0.30, 0.08)
    assert len(found) == 1  # only one more 8% stock buy fits under the 40% total
