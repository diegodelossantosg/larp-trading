import numpy as np
import pandas as pd

from trader.intraday_candles import cost_rt, run, same_time_controls, symbol_trades


def _walk(seed, days=120, drift=0.0):
    rng = np.random.default_rng(seed)
    frames = []
    p = 100.0
    for d in pd.bdate_range("2024-01-02", periods=days):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", tz="America/New_York")
        r = rng.normal(drift, 0.001, 78)
        c = p * np.exp(np.cumsum(r))
        o = np.r_[p, c[:-1]]
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0005, 78)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0005, 78)))
        frames.append(pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                                    "volume": rng.integers(1000, 5000, 78).astype(float)}, index=idx))
        p = c[-1]
    return pd.concat(frames)


def test_costs_are_per_symbol():
    assert cost_rt("SPY") < cost_rt("IWM") < cost_rt("AAPL")
    assert abs(cost_rt("AAPL") - 0.001) < 1e-12


def test_controls_use_same_slot_other_day():
    day = np.repeat(np.arange(5), 4)
    slot = np.tile(np.arange(4), 5)
    ctl = same_time_controls(day, slot, np.array([2, 7]), 3, np.random.default_rng(0))
    assert (slot[ctl] == slot[[2, 7]][:, None]).all()
    assert (day[ctl] != day[[2, 7]][:, None]).all()


def test_one_hour_candles_now_produce_trades():
    df = _walk(1)
    from trader.candle_backtest import resample
    t = symbol_trades(resample(df, 60), 60, np.random.default_rng(0))
    assert sum(len(v["gross"]) for v in t.values()) > 0


def test_random_walks_pass_nothing():
    bars = {s: _walk(i) for i, s in enumerate(["SPY", "QQQ", "XLE", "XLF", "AAPL"])}
    rep = run(bars, timeframes={"15 min": 15, "1 hour": 60})
    assert rep["rules_tested"] > 0
    assert rep["passing"] == []
    assert rep["spy_hold_validation_on_1000"] is not None


def test_a_real_edge_passes_and_a_fake_one_fails():
    from trader.intraday_candles import evaluate
    rng = np.random.default_rng(3)
    days = [d.date() for d in pd.bdate_range("2024-01-02", periods=300)]

    def arrays(mean):
        d = np.array(days * 2)
        return {"date": d, "minute": np.full(len(d), 60), "gross": rng.normal(mean, 0.003, len(d)),
                "ctrl": rng.normal(0, 0.003, len(d))}
    trades = {("15 min", "hammer", "any", "hold 1 hour"): {s: arrays(0.003) for s in ["A", "B", "C", "D", "E", "F"]},
              ("15 min", "hanging_man", "any", "hold 1 hour"): {s: arrays(0.0) for s in ["A", "B", "C", "D", "E", "F"]}}
    rep = evaluate(trades, None, rng)
    names = [r["pattern"] for r in rep["passing"]]
    assert names == ["hammer"]
    assert rep["passing"][0]["validation"]["pot_end_on_1000"] > 1000
