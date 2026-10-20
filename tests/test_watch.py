from trader.watch import Quote, buy_candidates, buy_orders, score, stop_orders, update_highs


def pos(value, entry, price):
    return {"market_value": value, "avg_entry_price": entry, "current_price": price}


def strong(**kw):
    """An ETF where all six buy signals agree."""
    base = dict(price=103.0, prev_close=100.0, high_20d=102.0, atr_pct=1.5, sma20=98.0, sma50=95.0,
                vwap=101.5, candles=["bullish_engulfing"], volume_ratio=2.0, bars_today=6)
    base.update(kw)
    return Quote(**base)


SPY_FLAT = Quote(500.5, 500, 510, 1, bars_today=6)


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


def test_score_counts_each_signal():
    assert score(strong())[0] == 6
    assert score(strong(candles=[]))[0] == 5
    assert score(strong(candles=[], volume_ratio=1.0, vwap=110))[0] == 3
    assert score(strong(price=108))[0] == 5  # up 8%: too extended to count as a healthy rise


def test_buy_candidates_need_min_score_calm_market_and_settled_open():
    quotes = {"SPY": SPY_FLAT, "SMH": strong(), "XLE": strong(candles=[], volume_ratio=1.0),
              "GLD": strong(bars_today=1)}
    assert [c[0] for c in buy_candidates(quotes, {}, "SPY", 5, 0.30, 0.08)] == ["SMH"]
    quotes["SPY"] = Quote(495, 500, 510, 1, bars_today=6)  # market falling
    assert buy_candidates(quotes, {}, "SPY", 5, 0.30, 0.08) == []


def test_buy_candidates_respect_position_cap():
    quotes = {"SPY": SPY_FLAT, "SMH": strong()}
    assert buy_candidates(quotes, {"SMH": 0.25}, "SPY", 5, 0.30, 0.08) == []
    assert buy_candidates(quotes, {"SMH": 0.15}, "SPY", 5, 0.30, 0.08)[0][0] == "SMH"


def test_buys_funded_from_spy():
    orders = buy_orders([("SMH", 6, ["x"])], equity=1000, cash=10, idle_value=500, idle_asset="SPY",
                        held={"SPY", "XLE"}, max_positions=8, size=0.08, slots=4)
    assert [(o.symbol, o.side, o.notional) for o in orders] == [("SPY", "sell", 72.0), ("SMH", "buy", 80.0)]
