import numpy as np
import pandas as pd

from trader import exit_backtest as eb
from trader import signal_backtest as sb
from tests.test_signal_backtest import _data


def _daily(seed, n=400, drift=0.0005):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    o = np.r_[100, c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c},
                      index=pd.bdate_range("2020-01-01", periods=n))
    return eb.Daily(df)


def test_daily_exit_stop_and_hold():
    df = pd.DataFrame({"open": [100, 100, 90, 90], "high": [100, 101, 91, 91], "low": [100, 99, 85, 89],
                       "close": [100, 100, 90, 90]}, index=pd.bdate_range("2020-01-01", periods=4))
    px, spy = eb.Daily(df), eb.Daily(df.assign(low=df["close"], high=df["close"]))
    r = eb.daily_exit(px, spy, 0, 3, 0.05)
    assert abs(r - (min(101 * 0.95, 90) / 100 - 0.9 - eb.COST)) < 1e-9   # stopped on day 2
    assert eb.daily_exit(px, spy, 0, 5, 0.05) is None                     # not enough days left


def test_grids_and_entries_run():
    spy = _daily(0)
    stocks = {f"S{i}": _daily(i + 1) for i in range(6)}
    drops = eb.drop_entries(stocks, spy)
    assert drops and all(len(e) == 3 for e in drops)
    rows = eb.grid(drops, (1, 3), lambda k, h, s: eb.daily_exit(stocks[k[0]], spy, k[1], h, s))
    assert rows and {"keep_days", "stop", "median_year_pct", "years_beating_spy"} <= set(rows[0])
    assert any(r["stop"].startswith("smart") for r in rows)
    picks = eb.trend_entries(stocks, spy, 2)
    assert picks
    out = eb.summarize_grid(rows, (3, "10%"))
    assert out["best"] and "current_setting" in out


def test_survivor_checks_run():
    intraday, daily = _data(days=200)
    sig, meta = sb.build(intraday, daily)
    rep = eb.survivor_checks(sig, meta, sb.Exits(intraday, daily, meta))
    assert set(rep) == {"1_older_unseen_years", "2_nudged_settings_older_years", "3_by_year_and_etf",
                        "4_vs_random_uptrend_buys_older_years"}
