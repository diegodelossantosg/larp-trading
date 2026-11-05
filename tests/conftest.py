import pytest

from trader import earnings_calendar


@pytest.fixture(autouse=True)
def no_live_earnings_calendar(monkeypatch):
    """Tests never call Nasdaq; by default the calendar is 'unavailable' and Claude's dates are used."""
    monkeypatch.setattr(earnings_calendar, "upcoming", lambda today, days, fetch=None: None)


@pytest.fixture(autouse=True)
def no_live_corporate_actions(monkeypatch):
    """Tests never call Alpaca's corporate-actions API; by default there are none."""
    from trader import corp_actions
    monkeypatch.setattr(corp_actions, "fetch", lambda cfg, symbols, today, back=3, ahead=30: [])
