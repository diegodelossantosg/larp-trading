import numpy as np
import pandas as pd

from trader import audit_backtest as ab


def _df(seed, n=500, drift=0.0004, start="2018-01-02"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    o = np.r_[100, c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c},
                        index=pd.bdate_range(start, periods=n))


def test_grids_include_a_delisted_company():
    daily = {"SPY": _df(0), **{f"C{i}": _df(i + 1) for i in range(5)}, "GONE": _df(9, n=300, drift=-0.004)}
    out = ab.grids_for(daily, list(daily)[1:])
    assert out["companies_with_data"] == 6 and out["after_a_drop"]["best"]


def test_news_exits_from_entry_price():
    daily = {"SPY": _df(0), "NVDA": _df(1)}
    days = [str(d.date()) for d in daily["NVDA"].index[300:340]]
    events = [{"symbol": "NVDA", "day": d, "reaction": -0.03, "entry_price": float(daily["NVDA"].loc[d, "close"]),
               "spy_entry": float(daily["SPY"].loc[d, "close"])} for d in days]
    out = ab.news_exits(events, daily)
    assert out["events"] == 40 and out["best_by_average"]
