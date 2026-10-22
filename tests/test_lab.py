import numpy as np
import pandas as pd

from trader.lab import Lab, equal_weight, pick_backtest, stats


def _make(tmp_path, n_stocks=30, days=700, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=days)
    syms = [f"S{i:02d}" for i in range(n_stocks)] + ["SPY"]
    drift = np.r_[np.linspace(-0.0005, 0.0015, n_stocks), 0.0004]   # later stocks trend up more
    rets = rng.normal(drift, 0.01, (days, len(syms)))
    close = (100 * np.exp(np.cumsum(rets, axis=0))).astype("float32")
    close[400:, 0] = np.nan                                         # one company is delisted
    years = list(range(2016, 2019))
    member = np.zeros((len(years), len(syms)), bool)
    member[:, :n_stocks] = True
    p = tmp_path / "lab.npz"
    np.savez_compressed(p, dates=np.array([d.strftime("%Y-%m-%d") for d in dates]), symbols=np.array(syms),
                        groups=np.array(["A"] * len(syms)), years=np.array(years), member=member,
                        etfs=np.array(["SPY"]), open=close, close=close, low=close * 0.995, volume=close * 0 + 1e6)
    return Lab(str(p))


def test_momentum_picks_the_trending_stocks_and_beats_the_basket(tmp_path):
    lab = _make(tmp_path)
    mom = pick_backtest(lab, "mom_12_1", "A", n=5)
    ew = equal_weight(lab, "A")
    assert mom.loc["2017":].sum() > ew.loc["2017":].sum()
    s = stats(mom, pd.Series(np.nan_to_num(lab.ret[:, lab.col["SPY"]]), index=lab.dates))
    assert "yearly_pct" in s and np.isfinite(s["yearly_pct"])


def test_stop_and_delisting_do_not_break_accounting(tmp_path):
    lab = _make(tmp_path)
    r = pick_backtest(lab, "reversal_1m", "A", n=5, stop=0.05)
    assert np.isfinite(r.to_numpy()).all() and r.min() > -0.5
