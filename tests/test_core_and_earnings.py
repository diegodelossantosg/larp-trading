import numpy as np
import pandas as pd

from trader import core_backtest as cb
from trader import earnings_backtest as ebt
from trader import exit_backtest as eb


def _df(seed, n=900, drift=0.0004, start="2017-01-02"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    o = np.r_[100, c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c},
                        index=pd.bdate_range(start, periods=n))


def test_core_policies_run_and_rank():
    spy = _df(0)
    daily = {"SPY": spy, "SSO": spy.assign(close=spy["close"] ** 2 / 100), **{f"C{i}": _df(i + 1) for i in range(8)}}
    out = cb.run(daily, [f"C{i}" for i in range(8)])
    rows = out["all_combinations"]
    assert len(rows) == 2 * 5 * 3 + 1
    assert {"core", "core_pct", "rule", "yearly_pct", "worst_drop_pct", "years_beating_spy"} <= set(rows[0])


def test_earnings_events_from_headlines():
    spy, px = _df(0, n=300), _df(1, n=300)
    daily = {"SPY": eb.Daily(spy), "NVDA": eb.Daily(px)}
    day = px.index[100]
    news = [{"time": f"{day.date()}T21:05:00Z", "symbols": ["NVDA"],
             "headline": "NVIDIA Q2 2025 EPS $0.68 Beats $0.64 Estimate, Sales $30.04B Beat $28.70B"},
            {"time": f"{day.date()}T22:00:00Z", "symbols": ["NVDA"], "headline": "NVIDIA EPS beats again"},  # same quarter
            {"time": f"{day.date()}T15:00:00Z", "symbols": ["NVDA"], "headline": "Analyst raises target"}]
    evs = ebt.earnings_events(news, daily, daily["SPY"])
    assert len(evs) == 1 and evs[0]["beat"] and evs[0]["day"] == str(px.index[101].date())  # reacts next session
