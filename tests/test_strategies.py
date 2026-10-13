import numpy as np
import pandas as pd

from trader.strategy_backtest import OTHERS, SECTORS, evaluate, run_weights


def test_run_weights_lags_signal_and_charges_costs():
    idx = pd.bdate_range("2024-01-01", periods=4)
    rets = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.10]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    r = run_weights(rets, w)
    assert r.iloc[0] == 0 and round(r.iloc[1], 4) == round(0.10 - 0.0005, 4) and r.iloc[2] == 0.10


def test_evaluate_runs_on_synthetic_prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    px = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))) for s in SECTORS + OTHERS}, index=idx)
    px["BIL"] = 100 * np.exp(np.arange(len(idx)) * 0.0001)
    res = evaluate(px)
    assert set(res["results"]) >= {"spy_hold", "sso_trend", "sector_momentum", "dual_momentum"}
    assert res["results"]["spy_hold"]["years_beating_spy"].startswith("0/")
