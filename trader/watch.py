"""The 10-minute watcher: fast, rules-only reactions between Claude's daily decisions.

1. Trailing stop: sell any position (other than the idle asset) that has fallen
   STOP_TRAIL_PCT below its highest price since it was bought.
2. Breakouts: buy an ETF that is rising sharply today (at least BREAKOUT_MIN_PCT and
   1.5x its usual daily range), at a new 20-day high, while SPY is not falling.
   Each buy is BREAKOUT_SIZE of capital, funded from the idle asset or cash, at most
   MAX_BREAKOUTS_PER_DAY per day, and it gets the same trailing stop.

No Claude call is made here, so it is quick and free to run often.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .risk import Order
from .universe import CORE_ETFS, LEVERAGED

log = logging.getLogger(__name__)


@dataclass
class Quote:
    price: float  # latest trade
    prev_close: float  # yesterday's close
    high_20d: float  # highest close of the previous 20 sessions
    atr_pct: float  # typical daily range, % of price


def update_highs(highs: dict[str, float], positions: dict[str, dict], quotes: dict[str, Quote]) -> dict[str, float]:
    """Highest price seen since each position was opened (positions no longer held are dropped)."""
    out = {}
    for sym, p in positions.items():
        q = quotes.get(sym)
        price = q.price if q else p["current_price"]
        out[sym] = max(highs.get(sym, 0.0), p["avg_entry_price"], price)
    return out


def stop_orders(positions: dict[str, dict], highs: dict[str, float], quotes: dict[str, Quote],
                idle_asset: str, trail_pct: float) -> list[Order]:
    orders = []
    for sym, p in positions.items():
        if sym == idle_asset or sym not in quotes:
            continue
        price, high = quotes[sym].price, highs.get(sym, p["avg_entry_price"])
        if price <= high * (1 - trail_pct):
            drop = 1 - price / high
            orders.append(Order(sym, "sell", round(p["market_value"], 2),
                                f"trailing stop: {drop:.1%} below its high of ${high:.2f}"))
    return orders


def breakout_candidates(quotes: dict[str, Quote], held: set[str], idle_asset: str,
                        min_pct: float) -> list[tuple[str, float]]:
    """(symbol, today's % change) for qualifying breakouts, strongest first."""
    spy = quotes.get("SPY")
    if spy is None or spy.price / spy.prev_close - 1 < -0.005:
        return []  # don't chase rallies on a falling market
    found = []
    for sym, q in quotes.items():
        if sym in held or sym == idle_asset or sym in LEVERAGED or sym not in CORE_ETFS:
            continue
        change = q.price / q.prev_close - 1
        if change >= max(min_pct, 1.5 * q.atr_pct / 100) and q.price >= q.high_20d:
            found.append((sym, change))
    return sorted(found, key=lambda x: -x[1])


def breakout_orders(candidates: list[tuple[str, float]], *, equity: float, cash: float,
                    idle_value: float, idle_asset: str, n_positions: int, max_positions: int,
                    size: float, slots: int) -> list[Order]:
    """Buys for the best candidates, plus the idle-asset sale that pays for them."""
    buys: list[Order] = []
    for sym, change in candidates[:slots]:
        if n_positions + len(buys) >= max_positions:
            break
        buys.append(Order(sym, "buy", round(size * equity, 2), f"breakout: up {change:.1%} today, new 20-day high"))
    need = sum(o.notional for o in buys) - max(0.0, cash - 0.01 * equity)
    if need > 0:
        sell = min(need, idle_value)
        if sell < need:  # can't fully fund: shrink the buys to what is available
            scale = (sum(o.notional for o in buys) - (need - sell)) / sum(o.notional for o in buys)
            buys = [Order(o.symbol, o.side, round(o.notional * scale, 2), o.reason) for o in buys if o.notional * scale >= 5]
        if sell >= 5 and buys:
            return [Order(idle_asset, "sell", round(sell, 2), "funding breakout buys")] + buys
    return buys
