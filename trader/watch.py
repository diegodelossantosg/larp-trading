"""The 5-minute watcher: fast, rules-only reactions between Claude's daily decisions.

1. Trailing stop: sell any position (other than the idle asset) that has fallen
   STOP_TRAIL_PCT (STOP_TRAIL_PCT_STOCK for single stocks) below its highest price since it was bought.
2. Intraday buys: score every ETF and big company stock on six signals from its daily trend and today's
   15-minute candles. Buy when at least WATCH_MIN_SCORE agree, the market (SPY) is not
   falling and the first half hour of the session is over. Each buy is WATCH_BUY_SIZE of
   capital, funded from the idle asset or cash, at most WATCH_MAX_BUYS_PER_DAY per day,
   and gets the same trailing stop.

No Claude call is made here, so it is quick and free to run often.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .risk import Order
from .universe import BIG_STOCKS, CORE_ETFS, LEVERAGED

log = logging.getLogger(__name__)

BULLISH_CANDLES = {"hammer", "inverted_hammer", "bullish_engulfing", "morning_star", "three_white_soldiers"}


@dataclass
class Quote:
    price: float  # latest trade
    prev_close: float  # yesterday's close
    high_20d: float  # highest close of the previous 20 sessions
    atr_pct: float  # typical daily range, % of price
    sma20: float = 0.0  # daily moving averages (previous sessions)
    sma50: float = 0.0
    vwap: float = 0.0  # today's volume-weighted average price
    candles: list[str] = field(default_factory=list)  # patterns on the latest 15-minute bars
    volume_ratio: float = 0.0  # latest 15-minute volume vs today's average bar
    bars_today: int = 0  # completed 15-minute bars so far today


def update_highs(highs: dict[str, float], positions: dict[str, dict], quotes: dict[str, Quote]) -> dict[str, float]:
    """Highest price seen since each position was opened (positions no longer held are dropped)."""
    out = {}
    for sym, p in positions.items():
        q = quotes.get(sym)
        price = q.price if q else p["current_price"]
        out[sym] = max(highs.get(sym, 0.0), p["avg_entry_price"], price)
    return out


def stop_orders(positions: dict[str, dict], highs: dict[str, float], quotes: dict[str, Quote],
                idle_asset: str, trail_pct: float, stocks: set[str] | frozenset = frozenset(),
                stock_trail_pct: float | None = None) -> list[Order]:
    orders = []
    for sym, p in positions.items():
        if sym == idle_asset or sym not in quotes:
            continue
        price, high = quotes[sym].price, highs.get(sym, p["avg_entry_price"])
        trail = stock_trail_pct if sym in stocks and stock_trail_pct else trail_pct
        if price <= high * (1 - trail):
            drop = 1 - price / high
            orders.append(Order(sym, "sell", round(p["market_value"], 2),
                                f"trailing stop: {drop:.1%} below its high of ${high:.2f}"))
    return orders


def score(q: Quote) -> tuple[int, list[str]]:
    """How many of the six buy signals agree, and which ones."""
    change = q.price / q.prev_close - 1
    signals = {
        "daily uptrend": q.sma20 > 0 and q.price > q.sma20 > q.sma50,
        f"up {change:.1%} today": max(0.01, 0.75 * q.atr_pct / 100) <= change <= 0.06,
        "above today's average price (VWAP)": q.vwap > 0 and q.price > q.vwap,
        "bullish candles: " + ", ".join(sorted(BULLISH_CANDLES & set(q.candles))): bool(BULLISH_CANDLES & set(q.candles)),
        f"volume {q.volume_ratio:.1f}x normal": q.volume_ratio >= 1.5,
        "new 20-day high": q.price >= q.high_20d,
    }
    hits = [name for name, ok in signals.items() if ok]
    return len(hits), hits


def buy_candidates(quotes: dict[str, Quote], weights: dict[str, float], idle_asset: str,
                   min_score: int, max_weight: float, size: float, *, max_stock_weight: float = 0.10,
                   max_stocks_total: float = 0.40, stocks: set[str] | frozenset = frozenset()
                   ) -> list[tuple[str, int, list[str]]]:
    """(symbol, score, reasons) for ETFs and big company stocks worth buying now, best first.

    `stocks` lists every single-company stock (held or scanned) so the 40% total for stocks holds.
    """
    stocks = set(stocks) | set(BIG_STOCKS)
    spy = quotes.get("SPY")
    if spy is None or spy.price / spy.prev_close - 1 < -0.005:
        return []  # don't chase rallies on a falling market
    found = []
    for sym, q in quotes.items():
        if sym == idle_asset or sym in LEVERAGED or (sym not in CORE_ETFS and sym not in BIG_STOCKS):
            continue
        cap = max_stock_weight if sym in stocks else max_weight
        if q.bars_today < 2 or weights.get(sym, 0.0) + size > cap + 1e-9:
            continue  # skip the opening half hour, and positions already at their cap
        s, reasons = score(q)
        if s >= min_score:
            found.append((sym, s, reasons))
    found.sort(key=lambda x: (-x[1], -(quotes[x[0]].price / quotes[x[0]].prev_close)))
    stock_total = sum(w for s, w in weights.items() if s in stocks)
    out = []
    for c in found:
        if c[0] in stocks:
            if stock_total + size > max_stocks_total + 1e-9:
                continue
            stock_total += size
        out.append(c)
    return out


def buy_orders(candidates: list[tuple[str, int, list[str]]], *, equity: float, cash: float,
               idle_value: float, idle_asset: str, held: set[str], max_positions: int,
               size: float, slots: int) -> list[Order]:
    """Buys for the best candidates, plus the idle-asset sale that pays for them."""
    buys: list[Order] = []
    n_positions = len(held - {idle_asset})
    for sym, s, reasons in candidates[:slots]:
        if sym not in held and n_positions >= max_positions:
            continue
        if sym not in held:
            n_positions += 1
        buys.append(Order(sym, "buy", round(size * equity, 2), f"{s}/6 signals: " + "; ".join(reasons)))
    return fund_buys(buys, equity=equity, cash=cash, idle_value=idle_value, idle_asset=idle_asset)


def fund_buys(buys: list[Order], *, equity: float, cash: float, idle_value: float, idle_asset: str) -> list[Order]:
    """The buys plus the idle-asset sale that pays for them (buys shrink if there isn't enough)."""
    if not buys:
        return []
    need = sum(o.notional for o in buys) - max(0.0, cash - 0.01 * equity)
    if need <= 0:
        return buys
    sell = min(need, idle_value)
    if sell < need:  # can't fully fund: shrink the buys to what is available
        total = sum(o.notional for o in buys)
        scale = (total - (need - sell)) / total
        buys = [Order(o.symbol, o.side, round(o.notional * scale, 2), o.reason) for o in buys if o.notional * scale >= 5]
    if sell >= 5 and buys:
        return [Order(idle_asset, "sell", round(sell, 2), "funding the buys")] + buys
    return buys
