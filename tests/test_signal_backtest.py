import numpy as np
import pandas as pd

from trader.signal_backtest import Exits, build, candidate_rules, run


def _data(seed=0, days=160):
    rng = np.random.default_rng(seed)
    intraday, daily = {}, {}
    for sym, drift in (("SPY", 0.0003), ("XLE", 0.001), ("XLK", 0.0)):
        frames, rows, price = [], [], 100.0
        for d in pd.bdate_range("2025-03-03", periods=days):
            idx = pd.date_range(f"{d.date()} 09:30", periods=26, freq="15min", tz="America/New_York")
            p = price * np.exp(np.cumsum(rng.normal(drift / 26, 0.003, 26)))
            o = np.r_[price, p[:-1]]
            hi, lo = np.maximum(o, p) * 1.001, np.minimum(o, p) * 0.999
            v = rng.integers(1000, 5000, 26).astype(float)
            frames.append(pd.DataFrame({"open": o, "high": hi, "low": lo, "close": p, "volume": v, "vwap": (o + p) / 2},
                                       index=idx))
            rows.append({"open": o[0], "high": hi.max(), "low": lo.min(), "close": p[-1], "volume": v.sum()})
            price = p[-1]
        intraday[sym] = pd.concat(frames)
        daily[sym] = pd.DataFrame(rows, index=pd.bdate_range("2025-03-03", periods=days))
    return intraday, daily


def test_signal_audit_runs_end_to_end():
    intraday, daily = _data()
    sig, meta = build(intraday, daily)
    assert sig.shape[1] == 15 and len(meta) > 1000
    ex = Exits(intraday, daily, meta)
    r = ex.result(0, 3, 0.06)
    assert r is None or -0.5 < r < 0.5
    rep = run(sig, meta, ex, min_train_trades=5, shortlist=5, top=3)
    assert rep["rules_tried"] > 10 and rep["best_by_first_year"]
    assert all("test" in r for r in rep["best_by_first_year"])
    assert len(candidate_rules()) == 587
