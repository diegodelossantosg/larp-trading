import numpy as np
import pandas as pd

from trader.overnight_backtest import daily_table, run


def _bars(seed, drift_night=0.0):
    rng = np.random.default_rng(seed)
    frames, price = [], 100.0
    for d in pd.bdate_range("2026-04-06", periods=60):
        price *= 1 + drift_night  # gap at the open
        idx = pd.date_range(f"{d.date()} 13:30", periods=78, freq="5min", tz="UTC")
        p = price * np.exp(np.cumsum(rng.normal(0, 0.0005, 78)))
        price = p[-1]
        frames.append(pd.DataFrame({"open": p, "high": p * 1.001, "low": p * 0.999, "close": p}, index=idx))
    return pd.concat(frames)


def test_daily_table_has_entry_and_next_morning_prices():
    t = daily_table(_bars(1))
    assert len(t) == 60
    assert t["n_open"].iloc[0] == t["open"].iloc[1]
    assert np.isnan(t["n_open"].iloc[-1])


def test_overnight_drift_is_found_and_scored():
    res = run({"SPY": _bars(1, drift_night=0.003), "XLE": _bars(2, drift_night=0.003)})
    assert res["combinations"] == 2 * 3 * 18
    best = res["strategies"][0]
    assert best["pot_end"] > 200 and best["avg_net_pct"] > 0
    assert res["spy_only_nights_no_costs"] > res["spy_only_daytime_no_costs"]
