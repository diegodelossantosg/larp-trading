import numpy as np
import pandas as pd

from trader.lab_data import group_of, yearly_universe


def test_yearly_universe_uses_that_januarys_volume(monkeypatch):
    idx = pd.bdate_range("2019-12-01", "2021-03-01")
    def bars(vol_2020, vol_2021):
        v = np.where(idx.year == 2020, vol_2020, vol_2021).astype("float32")
        return pd.DataFrame({"open": 10.0, "close": 10.0, "low": 9.5, "volume": v}, index=idx)
    b = {"AAA": bars(1e6, 1e3), "BBB": bars(1e3, 1e6), "PENNY": bars(1e9, 1e9).assign(close=1.0)}
    monkeypatch.setattr("trader.lab_data.TOP", 1)
    u = yearly_universe(b, range(2020, 2022))
    assert u == {2020: ["AAA"], 2021: ["BBB"]}


def test_groups_are_stable_and_three():
    assert group_of("AAPL") == group_of("AAPL")
    assert {group_of(s) for s in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]} == set("ABC")
