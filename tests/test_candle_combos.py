import numpy as np
import pandas as pd

from tests.test_candle_years import _bars
from trader import candle_combos as cc


def test_excess_return_is_stock_minus_spy_from_the_next_open():
    o = np.array([10.0, 10.0, 11.0, 12.0]); c = np.array([10.0, 10.5, 11.5, 12.5])
    so = np.array([100.0, 100.0, 100.0, 100.0]); sc = np.array([100.0, 101.0, 102.0, 103.0])
    r, ex = cc.excess_after(o, c, so, sc, 2)
    assert abs(r[0] - (12.5 / 10.0 - 1)) < 1e-12 or abs(r[0] - (11.5 / 10.0 - 1)) < 1e-12
    assert abs(r[0] - (c[2] / o[1] - 1)) < 1e-12 and abs(ex[0] - (r[0] - (sc[2] / so[1] - 1))) < 1e-12
    assert np.isnan(ex[3]) and np.isnan(ex[2])


def test_filters_are_boolean_masks_of_the_same_length():
    b = _bars(1, n=600)
    spy = np.ones(len(b), bool)
    m = cc.filter_masks(*(b[k].to_numpy() for k in ("open", "high", "low", "close", "volume")), spy)
    assert set(m) == set(cc.FILTERS)
    assert all(x.dtype == bool and len(x) == len(b) for x in m.values())
    assert not (m["market bull (SPY>200d)"] & m["market bear (SPY<200d)"]).any()


def test_analyze_on_random_prices_finds_nothing_that_passes_validation():
    bars = {f"S{i}": _bars(i, n=2900) for i in range(80)}
    spy = _bars(999, n=2900)
    years = list(range(2016, 2023))
    member = {y: set(bars) for y in years}
    bars_idx = pd.bdate_range("2014-01-02", periods=2900)
    for b in list(bars.values()) + [spy]:
        b.index = bars_idx
    res = cc.analyze(bars, spy, member, years)
    assert res["combinations_judged"] > 50
    assert res["passing_validation"] == []          # pure noise must not produce a "works"
