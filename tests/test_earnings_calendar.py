"""Nasdaq's earnings calendar decides the earnings rule when it is available."""

import json

import trader.earnings_calendar as ec
from trader.risk import earnings_blackout

from tests.test_news import idea, run

_real_upcoming = ec.upcoming   # conftest swaps it out during tests


def day(*symbols, time="time-after-hours"):
    return json.dumps({"data": {"rows": [{"symbol": s, "name": s, "time": time} for s in symbols]}}).encode()


def test_parse_reads_tickers_and_handles_empty_days():
    assert ec.parse(day("aapl ", "NVDA")) == {"AAPL", "NVDA"}
    assert ec.parse(json.dumps({"data": {"rows": None}}).encode()) == set()
    assert ec.parse(json.dumps({"data": None}).encode()) == set()


def test_reports_from_before_todays_open_are_already_out():
    assert ec.parse(day("MMM", time="time-pre-market"), skip_pre_market=True) == set()
    assert ec.parse(day("NFLX"), skip_pre_market=True) == {"NFLX"}                       # reports tonight
    assert ec.parse(day("XYZ", time="time-not-supplied"), skip_pre_market=True) == {"XYZ"}  # unclear: stay safe


def test_upcoming_covers_the_window_in_trading_days():
    seen = []
    cal = {"2026-10-16": {"AAPL"}, "2026-10-19": {"NVDA"}, "2026-10-20": {"AAPL"}}

    def fetch(d, skip_pre_market):
        seen.append(d)
        assert skip_pre_market == (d == "2026-10-15")   # only today's morning reports are already out
        return cal.get(d, set())

    out = _real_upcoming("2026-10-15", 3, fetch)
    assert seen == ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"]   # skips the weekend
    assert out == {"AAPL": "2026-10-16", "NVDA": "2026-10-19"}                # earliest date wins


def test_upcoming_is_none_when_any_day_fails():
    def fetch(d, skip_pre_market):
        raise OSError("blocked")
    assert _real_upcoming("2026-10-15", 3, fetch) is None


def test_calendar_overrides_claudes_dates():
    proposed = {"NVDA": 0.1, "AAPL": 0.1, "MSFT": 0.1}
    earnings = {"NVDA": "unknown", "AAPL": "2026-12-01", "MSFT": "unknown"}
    out, notes = earnings_blackout(proposed, {}, set(proposed), earnings, "2026-10-15", 3, {"AAPL": "2026-10-16"})
    assert out == {"NVDA": 0.1, "AAPL": 0.0, "MSFT": 0.1}   # unknown is fine if Nasdaq says no report soon
    assert notes == ["didn't add to AAPL: it reports earnings on 2026-10-16, too soon to buy"]


def test_news_hunter_uses_the_calendar(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "upcoming", lambda today, days: {"AAPL": "2026-10-16"})
    broker, rec = run(tmp_path, monkeypatch, [idea("NVDA", "unknown"), idea("AAPL", "2026-12-01")])
    assert [o.symbol for o in broker.sent if o.side == "buy"] == ["NVDA"]
    assert "skipped AAPL: it reports earnings on 2026-10-16, too soon to buy" in rec["notes"]
    assert "earnings dates checked on Nasdaq's calendar" in rec["notes"]
