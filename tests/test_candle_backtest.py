import numpy as np
import pandas as pd

from trader.candle_backtest import detect, intraday_trades, run_intraday, summarize_daily, BEAR


def _arr(rows):
    return [np.array(x, float) for x in zip(*rows)]


def test_detects_hammer_after_a_fall_and_engulfing():
    rows = [(10 - i * 0.2, 10.1 - i * 0.2, 9.7 - i * 0.2, 9.8 - i * 0.2) for i in range(8)]   # falling
    rows.append((8.2, 8.25, 7.0, 8.3 - 0.05))        # hammer: long lower wick, small body
    o, h, l, c = _arr(rows)
    c[-1] = 8.25; h[-1] = 8.27
    m = detect(o, h, l, c)
    assert m["hammer"][0][-1] and not m["shooting_star"][0][-1]
    rows = [(10 - i * 0.2, 10.1 - i * 0.2, 9.7 - i * 0.2, 9.8 - i * 0.2) for i in range(9)]
    rows.append((8.1, 8.9, 8.0, 8.8))                 # big up candle swallowing the previous down candle
    o, h, l, c = _arr(rows)
    assert detect(o, h, l, c)["bullish_engulfing"][0][-1]


def test_two_to_one_target_and_stop():
    # entry at 100 (bar 1 open), stop 99 -> target 102; bar 2 reaches 102.5
    o = np.array([100, 100, 101, 101.0]); h = np.array([100.5, 101, 102.5, 101.5])
    l = np.array([99, 99.5, 100.5, 100.0]); c = np.array([100, 100.8, 102, 101.0])
    r = intraday_trades(o, h, l, c, np.array([3, 3, 3, 3]), np.array([0]), np.array([99.0]), False, 2)
    assert abs(r["2:1 target"][0] - (0.02 - 0.001)) < 1e-9
    assert abs(r["hold to the close"][0] - (0.01 - 0.001)) < 1e-9
    # short: stop 100.6 above entry 100 is hit on bar 1 (high 101)
    r = intraday_trades(o, h, l, c, np.array([3, 3, 3, 3]), np.array([0]), np.array([100.6]), True, 2)
    assert abs(r["2:1 target"][0] - (-(100.6 / 100 - 1) - 0.001)) < 1e-9


def test_runs_end_to_end_on_random_walks():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-02 09:30", periods=78, freq="5min", tz="America/New_York")
    days = pd.bdate_range("2025-01-02", periods=60)
    bars = {}
    for sym in ("AAA", "SPY"):
        frames = []
        p = 100.0
        for d in days:
            t = idx + (d.normalize().tz_localize("America/New_York") - idx[0].normalize())
            steps = rng.normal(0, 0.002, 78)
            c = p * np.exp(np.cumsum(steps)); o = np.r_[p, c[:-1]]
            h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, 78)); l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, 78))
            frames.append(pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.integers(1e3, 1e4, 78)}, index=t.tz_convert("UTC")))
            p = c[-1]
        bars[sym] = pd.concat(frames)
    rep = run_intraday(bars)
    assert rep["5 min"]["combinations_tested"] > 0 and "1 hour" in rep


def test_bearish_sell_signal_scored_as_worse_than_random():
    res = {("random days", "any", 5, "no stop"): {2020: [0.0] * 50, 2021: [0.0] * 50}}
    res[("bearish_engulfing", "any", 5, "no stop")] = {2020: [-0.01] * 60, 2021: [-0.01] * 60}
    out = summarize_daily(res)
    assert out["passing"] and out["passing"][0]["edge_vs_random_pct"] > 0
