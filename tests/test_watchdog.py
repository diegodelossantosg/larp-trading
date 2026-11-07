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
    assert missing_agents(at(9, 45), True, [], True, None) == []                   # too early for trade and news


def test_everything_missing_midday():
    assert missing_agents(at(11), True, [], False, None) == ["trade", "watch", "news"]


def test_nothing_missing_when_all_ran():
    assert missing_agents(at(11), True, [OK], True, 40) == []


def test_news_gap_too_long():
    assert missing_agents(at(13), True, [OK], True, 95) == ["news"]


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
