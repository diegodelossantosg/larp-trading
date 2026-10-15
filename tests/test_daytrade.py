import numpy as np
import pandas as pd

from trader.daytrade_backtest import run, simulate_day


def _day(date, prices):
    idx = pd.date_range(f"{date} 13:30", periods=len(prices), freq="5min", tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.001, "low": p * 0.999, "close": p}, index=idx)


def test_breakout_and_momentum_trade_a_rising_day():
    prices = np.linspace(100, 103, 78)
    out = simulate_day(_day("2026-07-01", prices), prev_close=99.0, sma20=98.0, spy_trend_up=True)
    assert out["orb30"] > 0 and out["orb30_trend"] == out["orb30"]
    assert out["morning_momentum"] > 0 and out["last_half_hour"] > 0
    assert "dip_buy" not in out


def test_stop_limits_the_loss():
    prices = np.r_[np.full(6, 100.0), [100.5], np.linspace(100.4, 90, 71)]
    out = simulate_day(_day("2026-07-01", prices), prev_close=99.0, sma20=None, spy_trend_up=False)
    assert -0.02 < out["orb30"] < 0
    assert "orb30_trend" not in out


def test_run_reports_costs_and_pot():
    rng = np.random.default_rng(2)
    bars = {}
    for sym in ("SPY", "XLE"):
        frames = [_day(d.date(), 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 78))))
                  for d in pd.bdate_range("2026-06-01", periods=30)]
        bars[sym] = pd.concat(frames)
    res = run(bars)
    base = res["strategies"]["baseline"]
    assert base["trades"] > 0 and base["avg_net_pct"] < base["avg_gross_pct"]
    assert res["spy_hold_end"] is not None
