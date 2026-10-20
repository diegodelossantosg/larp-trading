import pytest

from trader import earnings_calendar


@pytest.fixture(autouse=True)
def no_live_earnings_calendar(monkeypatch):
    """Tests never call Nasdaq; by default the calendar is 'unavailable' and Claude's dates are used."""
    monkeypatch.setattr(earnings_calendar, "upcoming", lambda today, days, fetch=None: None)
