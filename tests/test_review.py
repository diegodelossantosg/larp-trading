from trader import main
from trader.risk import Order
from trader.watch import Quote
from tests.test_trade_flow import cfg

QUOTES = {"SPY": Quote(500, 500, 510, 1), "SMH": Quote(103, 100, 102, 1.5), "XLE": Quote(103, 100, 102, 1.5)}
ORDERS = [Order("GLD", "sell", 90, "trailing stop: 6.2% below its high"),
          Order("SPY", "sell", 160, "funding the watcher's buys"),
          Order("SMH", "buy", 80, "6/6 signals"), Order("XLE", "buy", 80, "5/6 signals")]
CANDS = [("SMH", 6, ["a"]), ("XLE", 5, ["b"])]
ACCT = {"equity": 1000, "cash": 10}


def test_without_claude_only_stops_go_through(monkeypatch):
    monkeypatch.setattr(main, "claude_available", lambda c: False)
    kept, notes = main._review(cfg(), ORDERS, QUOTES, {}, ACCT, CANDS)
    assert [(o.symbol, o.side) for o in kept] == [("GLD", "sell")]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in notes[0]


def test_claude_approves_one_and_funding_shrinks(monkeypatch, tmp_path):
    monkeypatch.setattr(main.journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "claude_available", lambda c: True)
    monkeypatch.setattr(main, "review_buys", lambda c, ctx: ({"SMH": (True, "AI capex news"), "XLE": (False, "spike")}, None))
    kept, notes = main._review(cfg(), ORDERS, QUOTES, {}, ACCT, CANDS)
    assert [(o.symbol, o.side, o.notional) for o in kept] == [("GLD", "sell", 90), ("SPY", "sell", 80.0), ("SMH", "buy", 80)]
    assert any("rejected XLE" in n for n in notes)


def test_review_failure_skips_buys(monkeypatch, tmp_path):
    monkeypatch.setattr(main.journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "claude_available", lambda c: True)
    monkeypatch.setattr(main, "review_buys", lambda c, ctx: (None, "Claude review failed: TimeoutExpired"))
    kept, _ = main._review(cfg(), ORDERS, QUOTES, {}, ACCT, CANDS)
    assert [o.symbol for o in kept] == ["GLD"]
