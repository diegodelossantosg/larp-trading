import pandas as pd

from trader.indicators import candlestick_patterns, rsi, summarize


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def trend(n, start=100.0, step=1.0):
    rows = []
    price = start
    for _ in range(n):
        rows.append([price, price + 1.5, price - 0.5, price + step * 0.8, 1_000_000])
        price += step
    return rows


def test_bullish_engulfing_after_downtrend():
    rows = trend(6, start=110, step=-1)
    rows += [[104, 104.5, 102.5, 103, 1e6], [102.5, 105.5, 102, 105, 1e6]]
    assert "bullish_engulfing" in candlestick_patterns(bars(rows))


def test_bearish_engulfing():
    rows = trend(6) + [[106, 107.5, 105.8, 107, 1e6], [107.5, 107.8, 105, 105.5, 1e6]]
    assert "bearish_engulfing" in candlestick_patterns(bars(rows))


def test_hammer_in_downtrend():
    rows = trend(6, start=110, step=-1) + [[102.3, 103.4, 99, 103.3, 1e6]]
    assert "hammer" in candlestick_patterns(bars(rows))


def test_doji():
    rows = trend(5) + [[105, 107, 103, 105.05, 1e6]]
    assert "doji" in candlestick_patterns(bars(rows))


def test_three_white_soldiers():
    rows = [[100, 102, 99.8, 101.8, 1e6], [101.5, 103.5, 101.3, 103.3, 1e6], [103, 105, 102.8, 104.8, 1e6]]
    assert "three_white_soldiers" in candlestick_patterns(bars(rows))


def test_rsi_bounds():
    up = pd.Series(range(1, 60), dtype=float)
    assert rsi(up) == 100.0
    down = pd.Series(range(60, 1, -1), dtype=float)
    assert rsi(down) < 1


def test_summarize_keys_and_returns():
    s = summarize(bars(trend(250)))
    assert s["ret_1d"] > 0 and s["ret_60d"] > 0
    assert s["vs_sma200_pct"] > s["vs_sma50_pct"] > 0
    assert isinstance(s["candles"], list)
