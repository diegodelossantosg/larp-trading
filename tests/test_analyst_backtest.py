import numpy as np
import pandas as pd

from trader import exit_backtest as eb
from trader.analyst_backtest import classify, events, analyze


def test_classify_headlines():
    assert classify("Goldman Sachs Upgrades Apple to Buy, Raises Price Target to $250") == "upgrade"
    assert classify("Nike Downgraded to Neutral at Citi") == "downgrade"
    assert classify("JP Morgan Initiates Coverage On Lilly with Overweight Rating") == "new coverage, Buy"
    assert classify("Morgan Stanley Maintains Overweight on Apple, Raises Price Target to $260") == "price target raised"
    assert classify("Barclays Maintains Equal-Weight on Ford, Lowers Price Target to $11") == "price target cut"
    assert classify("Apple Q3 EPS $1.57 Beats $1.43 Estimate") is None


def test_events_skip_earnings_days_and_run_end_to_end():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2023-01-02", periods=400)
    daily = {}
    for s in ("SPY", "AAA", "BBB"):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
        daily[s] = eb.Daily(pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx))
    news = []
    for k in range(40, 360, 8):
        t = pd.Timestamp(idx[k]).tz_localize("America/New_York") + pd.Timedelta(hours=8)
        news.append({"time": t.isoformat(), "headline": "Goldman Sachs Upgrades AAA to Buy", "symbols": ["AAA"]})
    t = pd.Timestamp(idx[40]).tz_localize("America/New_York") + pd.Timedelta(hours=7)
    news.append({"time": t.isoformat(), "headline": "AAA Q1 EPS $1 Beats $0.9 Estimate", "symbols": ["AAA"]})
    evs = events(news, daily, daily["SPY"])
    assert evs and all(e["kind"] == "upgrade" for e in evs)
    assert str(idx[40].date()) not in {e["day"] for e in evs}          # earnings day left out
    rep = analyze(evs, daily, daily["SPY"], ["AAA", "BBB"])
    assert rep["by_kind"]["upgrade"]["events"] == len(evs)
