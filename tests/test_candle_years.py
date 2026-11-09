import numpy as np
import pandas as pd

from trader import candle_years as cy


def _bars(seed, n=900):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    openp = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def test_returns_after_uses_the_next_open_and_a_later_close():
    o = np.array([10.0, 11.0, 12.0, 13.0]); c = np.array([10.5, 11.5, 12.5, 13.5])
    r = cy.returns_after(o, c, 2)
    assert abs(r[0] - (12.5 / 11.0 - 1)) < 1e-12 and np.isnan(r[3]) and np.isnan(r[2])


def test_stop_target_hits_stop_first_and_target():
    o = np.array([10, 10, 10, 10, 10.0]); c = o.copy()
    h = np.array([10, 10.2, 11.5, 10, 10.0]); l = np.array([10, 9.9, 9.95, 10, 10.0])
    assert abs(cy.stop_target(o, h, l, c, 0, True, 9.5) - 0.1) < 1e-9          # entry 10, risk 0.5, target 11
    l2 = np.array([10, 9.4, 9.95, 10, 10.0])
    assert abs(cy.stop_target(o, h, l2, c, 0, True, 9.5) - (-0.05)) < 1e-9     # stop touched first
    assert cy.stop_target(o, h, l, c, 0, True, 10.5) is None                    # a stop above the entry is not a trade


def test_analyze_runs_and_returns_a_verdict_for_every_pattern():
    bars = {f"S{i}": _bars(i) for i in range(30)}
    years = [2020, 2021, 2022]
    member = {y: set(bars) for y in years}
    res = cy.analyze(bars, member, years)
    assert set(res["verdict"]) == set(cy.BULL + cy.BEAR)
    assert all("verdict" in v for v in res["verdict"].values())
    some = next(p for p, r in res["patterns"].items() if r["holds"]["5"]["total"].get("n", 0) > 0)
    t = res["patterns"][some]["holds"]["5"]["total"]
    assert {"n", "avg", "win", "years_judged", "edge_ctx", "t_edge"} <= set(t)
