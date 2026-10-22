import numpy as np
import pandas as pd

from trader.event_dataset import DailyV, Hourly, build, split_of


def _daily(rng, idx, drift=0.0):
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(idx))))
    return DailyV(pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6},
                               index=idx))


def _hourly(daily_df_close, idx):
    rows, times = [], []
    for d, c in zip(idx, daily_df_close):
        for hh in (4, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            times.append(pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(hours=hh))
            rows.append(c)
    return Hourly(pd.DataFrame({"open": rows}, index=pd.DatetimeIndex(times).tz_convert("UTC")))


def test_build_earnings_and_news_rows_with_entries():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2023-01-02", periods=300)
    daily = {"SPY": _daily(rng, idx), "AAA": _daily(rng, idx, 0.001)}
    hourly = {s: _hourly(d.c, idx) for s, d in daily.items()}
    t_earn = (pd.Timestamp(idx[100]).tz_localize("America/New_York") + pd.Timedelta(hours=7)).isoformat()
    t_news = (pd.Timestamp(idx[200]).tz_localize("America/New_York") + pd.Timedelta(hours=10)).isoformat()
    # make the news day a clear move
    hourly["AAA"].by_day[idx[200].date()] = [
        (t.time(), o * (1.05 if t.hour >= 11 else 1)) for t, o in
        [(pd.Timestamp(idx[200]) + pd.Timedelta(hours=h), daily["AAA"].c[199]) for h in (4, 8, 9, 10, 11, 12, 13, 14, 15, 16)]]
    news = [{"time": t_earn, "headline": "AAA Q1 EPS $1.10 Beats $1.00 Estimate, Sales $5B Beat", "symbols": ["AAA"]},
            {"time": t_news, "headline": "AAA Wins Big Government Contract", "symbols": ["AAA"]}]
    rows = build(news, daily, hourly, {"AAA": "dev"})
    kinds = {r["kind"]: r for r in rows}
    e = kinds["earnings"]
    assert e["beat"] == 1 and e["sales_beat"] == 1 and "pre8_r5" in e and "close_s20" in e
    n = kinds["news"]
    assert n["move_1h"] >= 0.04 and "hour_after_r10" in n and "pre8_r5" not in n


def test_split_is_stable():
    assert split_of("ABC") == split_of("ABC") and {split_of(s) for s in ("A", "B", "C", "D", "E", "F")} == {"dev", "holdout"}
