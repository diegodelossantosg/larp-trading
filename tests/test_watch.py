from trader.watch import Quote, breakout_candidates, breakout_orders, stop_orders, update_highs


def pos(value, entry, price):
    return {"market_value": value, "avg_entry_price": entry, "current_price": price}


def test_trailing_stop_sells_after_drop_from_high():
    positions = {"XLE": pos(94, 100, 94), "SPY": pos(500, 500, 400)}
    quotes = {"XLE": Quote(93.0, 95, 0, 1), "SPY": Quote(400, 500, 0, 1)}
    highs = update_highs({"XLE": 100.0}, positions, quotes)
    orders = stop_orders(positions, highs, quotes, "SPY", 0.06)
    assert [(o.symbol, o.side) for o in orders] == [("XLE", "sell")]  # SPY is never stopped


def test_no_stop_within_trail():
    positions = {"XLE": pos(97, 100, 97)}
    quotes = {"XLE": Quote(97.0, 98, 0, 1)}
    assert stop_orders(positions, update_highs({}, positions, quotes), quotes, "SPY", 0.06) == []


def test_breakout_needs_new_high_strength_and_calm_spy():
    quotes = {"SPY": Quote(501, 500, 510, 1), "SMH": Quote(103, 100, 102, 1.5),
              "XLE": Quote(101, 100, 100, 1.5), "GLD": Quote(103, 100, 110, 1.5)}
    assert [s for s, _ in breakout_candidates(quotes, set(), "SPY", 0.02)] == ["SMH"]
    quotes["SPY"] = Quote(495, 500, 510, 1)  # market falling: no chasing
    assert breakout_candidates(quotes, set(), "SPY", 0.02) == []


def test_breakout_funded_from_spy():
    orders = breakout_orders([("SMH", 0.03)], equity=1000, cash=10, idle_value=500, idle_asset="SPY",
                             n_positions=2, max_positions=8, size=0.10, slots=2)
    assert [(o.symbol, o.side, o.notional) for o in orders] == [("SPY", "sell", 100.0), ("SMH", "buy", 100.0)]
