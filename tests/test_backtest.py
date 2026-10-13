import pandas as pd

from trader.backtest import quote_at, run, simulate_exit


def test_simulate_exit_trailing_stop_and_time_limit():
    idx = pd.date_range("2026-01-05 14:30", periods=6, freq="1D", tz="UTC")
    up_then_drop = pd.Series([100, 105, 110, 103, 104, 106], index=idx)
    assert simulate_exit(up_then_drop, 0, 0.06, 10) == (3, "stop")  # 103 <= 110 * 0.94
    flat = pd.Series([100, 101, 100, 101, 100, 101], index=idx)
    assert simulate_exit(flat, 0, 0.06, 2) == (2, "time")


def _day_bars(day, prices, vol=1000):
    idx = pd.date_range(f"{day} 13:30", periods=len(prices), freq="15min", tz="UTC")
    p = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"open": p - 0.1, "high": p + 0.3, "low": p - 0.3, "close": p,
                         "volume": vol, "vwap": p - 0.05})


def _daily(start_price, n=80, step=0.2):
    idx = pd.date_range("2025-09-01", periods=n, freq="B", tz="UTC")
    c = pd.Series([start_price + i * step for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1e6})


def test_quote_at_builds_watcher_inputs():
    b = _day_bars("2026-01-02", [100, 101, 102, 103])
    q = quote_at({"prev_close": 99, "high_20d": 100, "atr_pct": 1, "sma20": 98, "sma50": 95}, b, 2)
    assert q.price == 102 and q.bars_today == 3 and q.vwap > 0


def test_run_end_to_end_on_synthetic_data():
    daily = {"SPY": _daily(500, step=0.1), "SMH": _daily(80)}
    last = daily["SMH"]["close"].iloc[-1]
    day = (daily["SMH"].index[-1] + pd.offsets.BDay(1)).date()
    smh = _day_bars(day, [last * 1.003, last * 1.008, last * 1.02, last * 1.025, last * 1.03, last * 1.031])
    smh.loc[smh.index[2], "volume"] = 5000
    spy = _day_bars(day, [508.0, 508.2, 508.4, 508.5, 508.6, 508.7])
    res = run({"SPY": spy, "SMH": smh}, daily, min_scores=(4,), horizons=(1,))
    r = res["score>=4, hold<=1d"]
    assert r["trades"] == 1 and r["avg_return_pct"] is not None
