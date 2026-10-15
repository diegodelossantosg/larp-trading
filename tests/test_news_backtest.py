import numpy as np
import pandas as pd

from trader.news_backtest import events, summarize


def _bars(prices_by_day):
    frames = []
    for day, prices in prices_by_day.items():
        idx = pd.date_range(f"{day} 09:30", periods=len(prices), freq="5min", tz="America/New_York")
        p = np.asarray(prices, dtype=float)
        frames.append(pd.DataFrame({"open": p, "close": p}, index=idx))
    return pd.concat(frames)


def test_news_event_measures_reaction_and_later_returns():
    days = [str(d.date()) for d in pd.bdate_range("2026-07-06", periods=8)]
    flat = {d: [100.0] * 78 for d in days}
    stock = {d: [100.0] * 78 for d in days}
    stock[days[0]] = [100.0] * 12 + [103.0] * 66          # jumps 3% at 10:30 on news
    for d in days[1:]:
        stock[d] = [105.0] * 78                            # keeps rising afterwards
    news = [{"time": f"{days[0]}T14:25:00Z", "headline": "Nvidia wins deal", "symbols": ["NVDA"]}]
    evs = events(news, {"NVDA": _bars(stock), "SPY": _bars(flat)})
    assert len(evs) == 1
    e = evs[0]
    assert abs(e["reaction"] - 0.03) < 1e-9
    assert e["1 day"] > 0.015                               # bought at 103, worth 105 next day
    rep = summarize(evs * 3)
    assert rep["rose 3% to 6%"]["events"] == 3
