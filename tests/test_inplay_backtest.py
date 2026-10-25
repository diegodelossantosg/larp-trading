from datetime import date

import numpy as np

from trader.inplay_backtest import report, simulate


def _bars(prices):
    """5-minute bars whose closes are `prices`; each bar opens at the previous close (the first at 100)."""
    c = np.asarray(prices, dtype=float)
    o = np.r_[100.0, c[:-1]]
    return o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c


def test_up_first_candle_buys_the_breakout_and_holds_to_the_close():
    o, h, l, c = _bars(np.r_[100.5, np.linspace(100.6, 104, 77)])
    d, entry, exit_ = simulate(o, h, l, c, 1, atr=5.0)
    assert d == 1 and entry == h[0] and exit_ == c[-1]


def test_down_first_candle_goes_short():
    o, h, l, c = _bars(np.r_[99.5, np.linspace(99.4, 96, 77)])
    d, entry, exit_ = simulate(o, h, l, c, 1, atr=5.0)
    assert d == -1 and entry == l[0] and exit_ < entry


def test_tight_stop_cuts_a_reversal():
    o, h, l, c = _bars(np.r_[100.5, 100.7, np.linspace(100.6, 95, 76)])
    d, entry, exit_ = simulate(o, h, l, c, 1, atr=5.0)
    assert d == 1 and exit_ <= entry - 0.5 + 1e-9 and exit_ > 98


def test_no_breakout_no_trade():
    o, h, l, c = _bars(np.r_[101, np.full(77, 100.0)])
    h[1] = o[1] = 100.5   # the second bar opens lower, so nothing ever reaches the first bar's high
    assert simulate(o, h, l, c, 1, atr=5.0) is None


def test_report_has_groups_costs_and_pots():
    trades = []
    for i in range(30):
        day = date(2024, 1, 2 + i % 28)
        for k in (5, 15, 30):
            for s in ("inplay", "random"):
                trades.append({"day": day, "sym": "X", "group": "ABC"[i % 3], "set": s, "k": k, "n_cand": 2,
                               "dir": 1 if i % 2 else -1, "gross": 0.004 if i % 2 else -0.001, "risk": 0.005})
    out = report(trades)
    v = out["variants"]["inplay_5min_both"]
    assert set(v["groups"]) == {"A", "B", "C"}
    assert v["by_cost"]["0.05%"]["avg_net_pct"] < v["by_cost"]["0.02%"]["avg_net_pct"]
    assert v["pot_1x_end"] > 0 and v["pot_paper_end"] > 0
