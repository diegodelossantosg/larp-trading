from datetime import datetime
from zoneinfo import ZoneInfo

from trader import journal, main, watchdog
from trader.watchdog import missing_agents, news_age_min, trade_done, watch_active
from tests.test_trade_flow import FakeBroker, FakeMarket, cfg, fake_decide

ET = ZoneInfo("America/New_York")
TODAY = "2026-11-09"


def at(h, m=0):
    return datetime(2026, 11, 9, h, m, tzinfo=ET)


OK = {"date": TODAY, "market_view": "x", "orders": []}


def test_trade_done_cases():
    assert not trade_done([], TODAY)
    assert trade_done([OK], TODAY)
    assert not trade_done([{**OK, "date": "2026-11-06"}], TODAY)                   # yesterday doesn't count
    assert not trade_done([{**OK, "dry_run": True}], TODAY)                        # a dry run doesn't count
    assert not trade_done([{"date": TODAY, "source": "watch", "orders": []}], TODAY)
    assert trade_done([{"date": TODAY, "error": "HALTED: drawdown"}], TODAY)
    fail = {"date": TODAY, "error": "no decision"}
    assert not trade_done([fail, fail], TODAY)                                     # retry...
    assert trade_done([fail] * 3, TODAY)                                           # ...but not forever


def test_nothing_missing_when_market_closed():
    assert missing_agents(at(12), False, [], False, None) == []


def test_grace_period_after_the_open():
    assert missing_agents(at(9, 34), True, [], True, None) == []                   # too early for trade and news


def test_everything_missing_midday():
    assert missing_agents(at(11), True, [], False, None) == ["trade", "watch", "news"]


def test_nothing_missing_when_all_ran():
    assert missing_agents(at(11), True, [OK], True, 40) == []


def test_news_gap_too_long():
    assert missing_agents(at(13), True, [OK], True, 95) == ["news"]
    assert missing_agents(at(13), True, [OK], True, 50) == []                      # hourly: 50 minutes ago is fine
    assert missing_agents(at(13), True, [OK], True, 61) == ["news"]


def test_run_lists():
    assert watch_active([{"status": "completed"}, {"status": "in_progress"}])
    assert not watch_active([{"status": "completed"}])
    now = datetime(2026, 11, 9, 15, 0, tzinfo=ET)
    assert news_age_min([], now) is None
    assert news_age_min([{"createdAt": "2026-11-09T19:30:00Z"}], now) == 30       # 15:00 ET = 20:00 UTC


def test_trade_runs_once_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_today", lambda: TODAY)
    calls = []

    def decide(c, b):
        calls.append(1)
        return fake_decide({"market_view": "ok", "key_drivers": [], "trade_today": False, "confidence": "low",
                            "watchlist": [], "positions": []})(c, b)

    monkeypatch.setattr(main, "decide", decide)
    assert main.trade(cfg(), FakeBroker(), FakeMarket(), dry_run=False, force=False) == 0
    assert main.trade(cfg(), FakeBroker(), FakeMarket(), dry_run=False, force=False) == 0
    assert len(calls) == 1                                                          # second start did nothing
    main.trade(cfg(), FakeBroker(), FakeMarket(), dry_run=False, force=True)
    assert len(calls) == 2                                                          # --force runs it again


def test_pending_sale_comes_first():
    assert missing_agents(at(11), True, [], False, None, liquidate_pending=True) == ["liquidate", "watch"]
    assert missing_agents(at(9, 35), True, [], True, None, liquidate_pending=True) == []


def test_trade_waits_for_the_pending_sale(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    journal.write("liquidate.json", {"pending": True})
    calls = []
    monkeypatch.setattr(main, "decide", lambda c, b: calls.append(1))
    assert main.trade(cfg(), FakeBroker(), FakeMarket(), dry_run=False, force=False) == 0
    assert not calls


def test_ideas_are_paper_only_by_default():
    from trader.main import ideas_to_real_money
    proposed = {"XLV": 0.2, "LLY": 0.1, "SPY": 0.5}
    out, note = ideas_to_real_money(proposed, "SPY", 0.0)
    assert out == {"SPY": 0.5} and "paper" in note
    out, note = ideas_to_real_money(proposed, "SPY", 0.15)          # scaled down to 15% of the money
    assert abs(out["XLV"] + out["LLY"] - 0.15) < 1e-3 and out["SPY"] == 0.5 and "15%" in note
    out, note = ideas_to_real_money(proposed, "SPY", 0.5)           # already under the limit: untouched
    assert out == proposed and note == ""


def test_report_card_counts_paper_ideas_as_picks():
    from trader.report_card import picks
    d = {"date": "2026-11-09", "market_view": "x", "ideas_paper_only": True, "orders": [],
         "positions": [{"symbol": "xlv", "trade_type": "macro_theme"}, {"symbol": "SPY"}]}
    assert [p["symbol"] for p in picks([d], "SPY")] == ["XLV"]


def test_default_run_puts_the_money_in_spy_and_only_records_ideas(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_today", lambda: TODAY)
    monkeypatch.setattr(main, "decide", fake_decide({
        "market_view": "x", "key_drivers": [], "trade_today": True, "confidence": "low", "watchlist": [],
        "positions": [{"symbol": "XLE", "weight": 0.3, "thesis": "", "signals": [], "exit_plan": ""}]}))
    broker = FakeBroker()
    main.trade(cfg(ideas_share=0.0, idle_asset="SPY", max_idle_weight=1.0), broker, FakeMarket(), dry_run=False, force=False)
    rec = journal.read("decisions.json", [])[-1]
    assert rec["ideas_paper_only"] is True and rec["positions"][0]["symbol"] == "XLE"   # the idea is on record...
    assert {o.symbol for o in broker.sent} == {"SPY"}                                    # ...but only SPY is bought


def test_idle_choice_is_redone_when_the_setting_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_today", lambda: TODAY)
    journal.write("idle.json", {"date": TODAY, "asset": "SSO", "why": "old"})            # chosen today under the old settings

    class M:
        def daily_bars(self, symbols, days=400):
            raise RuntimeError("no data")

    c = main._with_todays_idle(cfg(idle_asset="SPY"), M())
    assert c.idle_asset == "SPY"                                                           # not the stale SSO
    assert journal.read("idle.json", {})["setting"] == "SPY/SPY"


def test_the_benchmark_starts_when_the_money_was_invested():
    state = {"capital": 1000.0, "baseline_equity": 99933.35, "started": "2026-11-09"}
    decisions = [{"date": "2026-11-09", "time": "11:02", "source": "news"},                  # not the daily decision
                 {"date": "2026-11-09", "time": "11:02", "dry_run": True},                   # a dry run doesn't count
                 {"date": "2026-11-09", "time": "11:02", "market_view": "x"}]
    assert main._started_at(state, decisions) == "2026-11-09T16:02+00:00"                    # 11:02 New York = 16:02 UTC
    assert main._started_at(state, [{"date": "2026-11-06", "time": "11:02"}]) is None


def test_the_benchmark_uses_the_price_the_spy_buy_got():
    fills = [{"symbol": "SPY", "side": "sell", "price": 740.0, "time": "2026-11-09T16:01:30+00:00"},
             {"symbol": "XLV", "side": "buy", "price": 150.0, "time": "2026-11-09T16:02:10+00:00"},
             {"symbol": "SPY", "side": "buy", "price": 750.62, "time": "2026-11-09T16:02:20+00:00"},
             {"symbol": "SPY", "side": "buy", "price": 755.0, "time": "2026-11-09T18:00:00+00:00"}]
    assert main._spy_fill_price(fills, "2026-11-09T16:02+00:00") == 750.62
    assert main._spy_fill_price(fills[:2], "2026-11-09T16:02+00:00") is None
