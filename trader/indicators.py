"""Technical indicators and candlestick patterns computed from daily OHLCV bars.

Pure functions over a pandas DataFrame with columns open, high, low, close, volume
(oldest row first). No network access, so this is fully unit-tested.
"""

from __future__ import annotations

import math

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain.iloc[-1] / last_loss))


def atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1] / df["close"].iloc[-1] * 100)


def macd_hist(close: pd.Series) -> float:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return float((macd - signal).iloc[-1])


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _range(row) -> float:
    return max(row["high"] - row["low"], 1e-12)


def _bull(row) -> bool:
    return row["close"] > row["open"]


def _bear(row) -> bool:
    return row["close"] < row["open"]


def candlestick_patterns(df: pd.DataFrame) -> list[str]:
    """Patterns completed on the most recent bar."""
    if len(df) < 3:
        return []
    c3, c2, c1 = df.iloc[-3], df.iloc[-2], df.iloc[-1]  # c1 is the latest bar
    found: list[str] = []

    body, rng = _body(c1), _range(c1)
    upper = c1["high"] - max(c1["open"], c1["close"])
    lower = min(c1["open"], c1["close"]) - c1["low"]
    downtrend = df["close"].iloc[-6:-1].mean() > c1["close"] if len(df) >= 6 else _bear(c2)
    uptrend = df["close"].iloc[-6:-1].mean() < c1["close"] if len(df) >= 6 else _bull(c2)

    if body <= 0.1 * rng:
        found.append("doji")
    elif lower >= 2 * body and upper <= 0.3 * body + 0.1 * rng:
        found.append("hammer" if downtrend else "hanging_man")
    elif upper >= 2 * body and lower <= 0.3 * body + 0.1 * rng:
        found.append("inverted_hammer" if downtrend else "shooting_star")

    if _bear(c2) and _bull(c1) and c1["open"] <= c2["close"] and c1["close"] >= c2["open"] and body > _body(c2):
        found.append("bullish_engulfing")
    if _bull(c2) and _bear(c1) and c1["open"] >= c2["close"] and c1["close"] <= c2["open"] and body > _body(c2):
        found.append("bearish_engulfing")

    small_middle = _body(c2) <= 0.3 * _body(c3)
    if _bear(c3) and small_middle and _bull(c1) and c1["close"] > (c3["open"] + c3["close"]) / 2:
        found.append("morning_star")
    if _bull(c3) and small_middle and _bear(c1) and c1["close"] < (c3["open"] + c3["close"]) / 2:
        found.append("evening_star")

    if all(_bull(c) for c in (c3, c2, c1)) and c3["close"] < c2["close"] < c1["close"]:
        found.append("three_white_soldiers")
    if all(_bear(c) for c in (c3, c2, c1)) and c3["close"] > c2["close"] > c1["close"]:
        found.append("three_black_crows")
    return found


def _ret(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return round(float(close.iloc[-1] / close.iloc[-1 - days] - 1) * 100, 2)


def summarize(df: pd.DataFrame) -> dict:
    """One compact row of signals for a symbol, for Claude to read."""
    close = df["close"]
    last = float(close.iloc[-1])
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else math.nan
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else math.nan
    vol20 = close.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252) * 100
    dollar_vol = float((df["close"] * df["volume"]).tail(20).mean())

    def pct_from(x: float) -> float | None:
        return None if math.isnan(x) else round((last / x - 1) * 100, 2)

    return {
        "price": round(last, 2),
        "ret_1d": _ret(close, 1),
        "ret_5d": _ret(close, 5),
        "ret_20d": _ret(close, 20),
        "ret_60d": _ret(close, 60),
        "vs_sma50_pct": pct_from(sma50),
        "vs_sma200_pct": pct_from(sma200),
        "rsi14": round(rsi(close), 1) if len(close) > 15 else None,
        "macd_hist": round(macd_hist(close), 3) if len(close) > 35 else None,
        "atr_pct": round(atr_pct(df), 2) if len(df) > 15 else None,
        "vol_ann_pct": None if math.isnan(vol20) else round(float(vol20), 1),
        "avg_dollar_vol_m": round(dollar_vol / 1e6, 1),
        "candles": candlestick_patterns(df),
    }
