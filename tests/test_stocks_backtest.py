import numpy as np
import pandas as pd

from trader.stocks_backtest import picker, run_picker


def _closes(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = {"SPY": 0.0004, "SSO": 0.0008, "XLE": 0.0003, "XLK": 0.0006, "GLD": 0.0001,
            "NVDA": 0.002, "AAPL": 0.0007, "KO": 0.0002}
    return pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(mu, 0.012, n))) for s, mu in cols.items()}, index=idx)


def test_picker_holds_half_in_sso_and_limits_stocks():
    closes = _closes()
    v = picker(closes, ["XLE", "XLK", "GLD", "NVDA", "AAPL", "KO"], {"NVDA", "AAPL", "KO"})
    assert len(v) == len(closes) and v.iloc[0] == 1000 and v.iloc[-1] > 0


def test_run_picker_reports_all_variants():
    res = run_picker(_closes())
    assert set(res["results"]) == {"old: ETFs only", "new: ETFs + big stocks",
                                   "new, without the 6 biggest winners", "just hold the S&P 500"}
    assert res["results"]["just hold the S&P 500"]["end_value"] > 0
