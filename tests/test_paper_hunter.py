from trader import paper_hunter as ph


def state():
    return ph.new_state(500.0, "2026-11-16")


def cand(sym, vs, price=100.0):
    return {"symbol": sym, "vs_spy": vs, "price": price, "why": f"{sym} news"}


def test_buys_biggest_fall_first_with_the_daily_and_total_limits():
    s = state()
    prices = {}
    bought = ph.pick_buys(s, [cand("A", -0.03), cand("B", -0.06), cand("C", -0.04)], prices, 500.0, "2026-11-16")
    assert [c["symbol"] for c in bought] == ["B", "C"]                          # 2 a day, biggest fall first
    assert abs(s["cash"] - 800.0) < 1e-6 and len(s["positions"]) == 2
    assert ph.pick_buys(s, [cand("D", -0.09)], {}, 500.0, "2026-11-16") == []   # already 2 today


def test_total_exposure_is_capped_at_30_percent():
    s = state()
    for day in ("2026-11-16", "2026-11-17"):
        ph.pick_buys(s, [cand(f"X{day}1", -0.05), cand(f"X{day}2", -0.04)], {}, 500.0, day)
    assert len(s["positions"]) == 3                                              # 3 x 10% = 30% and no more


def test_exit_after_ten_trading_days_and_on_the_stop():
    s = state()
    ph.pick_buys(s, [cand("A", -0.05, 100.0), cand("B", -0.04, 50.0)], {}, 500.0, "2026-11-02")
    closed = ph.apply_exits(s, {"A": 110.0, "B": 39.0}, 520.0, "2026-11-04")    # B fell 22% from 50: stop
    assert [c["symbol"] for c in closed] == ["B"] and "stop" in closed[0]["reason"]
    closed = ph.apply_exits(s, {"A": 110.0}, 520.0, "2026-11-16")               # 10 business days after Nov 2
    assert [c["symbol"] for c in closed] == ["A"] and closed[0]["return_pct"] == 10.0
    assert closed[0]["vs_spy_pct"] == 6.0                                        # +10% vs SPY +4%
    assert s["positions"] == []


def test_equity_counts_cash_and_open_positions():
    s = state()
    ph.pick_buys(s, [cand("A", -0.05, 100.0)], {}, 500.0, "2026-11-16")
    assert abs(ph.equity(s, {"A": 100.0}) - (900.0 + 100.0 * 0.9995)) < 1e-6
    assert ph.equity(s, {"A": 120.0}) > 1000.0


def test_run_with_fake_market_buys_a_qualifying_drop_and_saves_state(tmp_path, monkeypatch):
    from trader import journal, main
    from trader.watch import Quote
    from tests.test_news import HeldBroker, NewsMarket
    from tests.test_trade_flow import cfg

    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "NEWS_MIN_ATR_PCT", 0.0)                                  # the fake history is calm
    monkeypatch.setattr(main, "_heavy_volume", lambda df: None)
    monkeypatch.setattr("trader.earnings_calendar.upcoming", lambda today, days, fetch=None: {})

    class M(NewsMarket):
        def quotes(self, symbols):
            return {s: (Quote(100.0, 100.0, 0, 1) if s == "SPY" else Quote(94.0, 100.0, 0, 1)) for s in symbols}

    class B(HeldBroker):
        def market_open(self):
            return True

    assert ph.run(cfg(), B(), M()) == 0
    st = journal.read("paper_hunter.json", None)
    assert st and st["curve"] and st["spy_start"] == 100.0
    assert st["positions"] and all(p["symbol"] in ("NVDA", "ACME") for p in st["positions"])
    assert len(st["positions"]) <= 2 and st["cash"] < 1000.0
