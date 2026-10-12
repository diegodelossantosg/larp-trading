import pytest

from trader.risk import apply_limits, drawdown, plan_orders

LIMITS = dict(idle_asset="SPY", max_position_weight=0.30, max_positions=3, max_leveraged_weight=0.10)


def test_remainder_goes_to_idle_asset():
    t, _ = apply_limits({"XLE": 0.2}, allowed={"XLE", "SPY"}, leveraged=set(), **LIMITS)
    assert t == {"XLE": 0.2, "SPY": 0.8}


def test_caps_unknown_symbols_and_position_count():
    t, notes = apply_limits(
        {"XLE": 0.5, "FAKE": 0.2, "A": 0.1, "B": 0.1, "C": 0.05},
        allowed={"XLE", "A", "B", "C", "SPY"}, leveraged=set(), **LIMITS,
    )
    assert t["XLE"] == 0.30
    assert "FAKE" not in t and "C" not in t
    assert sum(t.values()) == pytest.approx(1.0)
    assert any("FAKE" in n for n in notes)


def test_leveraged_cap():
    t, _ = apply_limits({"TQQQ": 0.3, "SQQQ": 0.1}, allowed={"TQQQ", "SQQQ", "SPY"},
                        leveraged={"TQQQ", "SQQQ"}, **LIMITS)
    assert t["TQQQ"] + t["SQQQ"] == pytest.approx(0.10)


def test_overallocation_is_scaled_down():
    t, _ = apply_limits({"A": 0.3, "B": 0.3, "C": 0.3, "SPY": 0.5}, allowed={"A", "B", "C", "SPY"},
                        leveraged=set(), **LIMITS)
    assert sum(t.values()) == pytest.approx(1.0)


def test_plan_orders_sells_first_and_respects_band():
    orders, _ = plan_orders(
        {"SPY": 0.5, "XLE": 0.5}, {"SPY": 6000, "QQQ": 4000}, 10_000,
        rebalance_band=0.02, min_order_usd=25, max_daily_turnover=1.0, cash_buffer=0.0,
    )
    sides = [(o.side, o.symbol) for o in orders]
    assert sides[0] == ("sell", "QQQ")
    assert ("buy", "XLE") in sides
    assert sides.index(("buy", "XLE")) > sides.index(("sell", "SPY"))


def test_small_drift_is_ignored():
    orders, _ = plan_orders({"SPY": 1.0}, {"SPY": 9900}, 10_000, rebalance_band=0.02,
                            min_order_usd=25, max_daily_turnover=1.0, cash_buffer=0.0)
    assert orders == []


def test_turnover_limit_scales_orders():
    orders, notes = plan_orders({"XLE": 1.0}, {"SPY": 10_000}, 10_000, rebalance_band=0.02,
                                min_order_usd=25, max_daily_turnover=0.25, cash_buffer=0.0)
    assert sum(o.notional for o in orders) == pytest.approx(5000)
    assert notes


def test_drawdown():
    assert drawdown([100, 120, 90]) == pytest.approx(0.25)
    assert drawdown([]) == 0.0
