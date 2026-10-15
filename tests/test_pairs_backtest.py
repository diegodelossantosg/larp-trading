import numpy as np
import pandas as pd

from trader import pairs_backtest as pb


def _closes(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    common = np.cumsum(rng.normal(0.0003, 0.01, n))
    data = {}
    for a, b in pb.PAIRS[:3]:
        spread = np.zeros(n)
        for t in range(1, n):  # mean-reverting gap between the twins
            spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.01)
        data[a] = 100 * np.exp(common + spread / 2)
        data[b] = 100 * np.exp(common - spread / 2)
    return pd.DataFrame(data, index=idx), pd.Series(100 * np.exp(common), index=idx)


def test_mean_reverting_twins_are_profitable_for_some_rule():
    closes, spy = _closes()
    rep = pb.evaluate(closes, spy)
    assert rep["rules_tested"] == 54
    best = rep["best_market_neutral_by_2016_2020"][0]
    assert best["all_years"]["yearly_pct"] > 0      # strong mean reversion: some pair rule profits here
    assert "yearly_pct" in rep["video_rule_daily"]["all_years"]


def test_costs_and_borrow_are_charged():
    r = pb._run(np.zeros(3), np.zeros(3), np.array([0.5, 0.5, 0.5]), np.array([-0.5, -0.5, -0.5]))
    assert r[0] < 0 and r[1] < 0 and r[0] < r[1]  # trading cost on day one, borrow every day
