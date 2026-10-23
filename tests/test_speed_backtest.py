import numpy as np
import pandas as pd

from trader.speed_backtest import sessions, simulate, summarize


def _bars(days, drop_day=None, drop_at=None, base=100.0):
    rows, idx = [], []
    for k, d in enumerate(days):
        for m in range(570, 960, 5):
            p = base
            if d == drop_day and m >= drop_at:
                p = base * 0.95
            idx.append(pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(minutes=m))
            rows.append((p, p))
    df = pd.DataFrame(rows, columns=["open", "close"], index=pd.DatetimeIndex(idx).tz_convert("UTC"))
    return sessions(df)


def test_faster_looks_buy_sooner_after_the_news():
    days = list(pd.bdate_range("2025-03-03", periods=20).date)
    d = days[3]
    bars = {"AAA": _bars(days, d, 600)}                          # drops 5% at 10:00
    spy_order, spy_days = _bars(days)
    news = {"AAA": [(598, d)]}                                   # headline 9:58
    fast = simulate(news, bars, spy_order, spy_days, {"AAA": "dev"}, 15)
    slow = simulate(news, bars, spy_order, spy_days, {"AAA": "dev"}, 60)
    assert len(fast) == len(slow) == 1
    assert fast[0]["look"] == 600 + 15 and slow[0]["look"] == 630   # first look after the headline
    assert fast[0]["minutes_after_news"] < slow[0]["minutes_after_news"]
    s = summarize(fast, 1.0)
    assert s["all"]["buys"] == 1 and "dollars_per_year_on_50_usd_trades" in s["all"]
