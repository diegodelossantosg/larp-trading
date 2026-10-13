import numpy as np
import pandas as pd

from trader.scalp_backtest import _exit, run


def test_exit_assumes_stop_when_both_touched():
    o = np.array([100.0, 100.0]); h = np.array([100.5, 100.5]); l = np.array([99.0, 99.0]); c = o
    assert _exit(o, h, l, c, 0, 100.2, 99.9, 1)[2] == "stop"


def test_exit_target_and_time():
    o = np.array([100.0, 100.1, 100.1]); h = np.array([100.1, 100.3, 100.2]); l = np.array([99.95, 100.0, 100.0])
    c = np.array([100.05, 100.2, 100.1])
    assert _exit(o, h, l, c, 0, 100.2, 99.8, 2)[2] == "target"
    assert _exit(o, h, l, c, 0, 101, 99, 2) == (100.1, 2, "time")


def test_run_on_random_walk_has_no_edge_after_costs():
    rng = np.random.default_rng(1)
    frames = []
    for d in pd.bdate_range("2026-07-01", periods=20):
        idx = pd.date_range(f"{d.date()} 13:30", periods=390, freq="1min", tz="UTC")
        p = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, 390)))
        frames.append(pd.DataFrame({"open": p, "high": p * 1.0003, "low": p * 0.9997, "close": p,
                                    "volume": rng.integers(1000, 5000, 390), "vwap": p}, index=idx))
    res = run({"XLE": pd.concat(frames)})
    for s in res["strategies"].values():
        if s["trades"]:
            assert s["avg_net_pct"] < s["avg_gross_pct"]  # costs always hurt
