"""Rule-only twin of the News Hunter, on paper. No Claude, no real money, never touches the Alpaca account.

    python -m trader.paper_hunter       # run once; the hourly News Hunter workflow calls it after the real one

Same plan as the audited News Hunter rule: a stock named in the last 2 hours of headlines that has fallen 2% or more
MORE than the S&P 500 today, volatile (daily swing 3%+), liquid, not on heavy volume, not about to report earnings.
Biggest fall first, 10% of the twin's money per buy, 2 buys a day, 30% in total. Out after 10 trading days or when the
price falls 20% from its best since bought. The result is compared with the same money left in SPY and with the real
News Hunter (where Claude picks), to see whether Claude's judgement adds anything. State: docs/data/paper_hunter.json.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from . import journal

log = logging.getLogger("paper_hunter")
ET = ZoneInfo("America/New_York")
FILE = "paper_hunter.json"
CAPITAL = 1000.0
SIZE, MAX_BUYS, CAP = 0.10, 2, 0.30
HOLD_DAYS, MIN_DROP, HALF_COST = 10, 0.02, 0.0005     # 0.05% each way = 0.1% round trip
STOP: float | None = None    # no trailing stop, like the real News Hunter buys (hour-by-hour replay)
CURVE_KEEP = 3000


def new_state(spy_price: float | None, today: str) -> dict:
    return {"started": today, "capital": CAPITAL, "cash": CAPITAL, "spy_start": spy_price, "positions": [], "closed": [],
            "buys": {}, "curve": []}


def equity(state: dict, prices: dict[str, float]) -> float:
    return state["cash"] + sum(p["shares"] * prices.get(p["symbol"], p["entry_price"]) for p in state["positions"])


def business_days(a: str, b: str) -> int:
    return int(np.busday_count(a, b))


def apply_exits(state: dict, prices: dict[str, float], spy_price: float | None, today: str) -> list[dict]:
    """Close positions that are 10 trading days old or 20% below their high; returns what was closed."""
    closed, keep = [], []
    for p in state["positions"]:
        px = prices.get(p["symbol"])
        if px is None:
            keep.append(p)
            continue
        p["high"] = max(p["high"], px)
        reason = ("stop: fell 20% from its best" if STOP and px <= p["high"] * (1 - STOP)
                  else f"{HOLD_DAYS} trading days" if business_days(p["entry_date"], today) >= HOLD_DAYS else None)
        if reason is None:
            keep.append(p)
            continue
        state["cash"] += p["shares"] * px * (1 - HALF_COST)
        ret = px / p["entry_price"] - 1
        spy_ret = (spy_price / p["spy_entry"] - 1) if spy_price and p.get("spy_entry") else None
        closed.append({"symbol": p["symbol"], "entry_date": p["entry_date"], "exit_date": today, "entry_price": p["entry_price"],
                       "exit_price": round(px, 4), "reason": reason, "return_pct": round(ret * 100, 2),
                       "vs_spy_pct": None if spy_ret is None else round((ret - spy_ret) * 100, 2), "why": p.get("why", "")})
    state["positions"] = keep
    state["closed"] += closed
    return closed


def pick_buys(state: dict, candidates: list[dict], prices: dict[str, float], spy_price: float | None, today: str) -> list[dict]:
    """candidates: [{symbol, vs_spy, price, why}] already vetted. Biggest fall first, within today's and the total limits."""
    eq = equity(state, prices)
    held = {p["symbol"] for p in state["positions"]}
    made = state["buys"].get(today, 0)
    bought = []
    for c in sorted(candidates, key=lambda c: c["vs_spy"]):
        invested = sum(p["shares"] * prices.get(p["symbol"], p["entry_price"]) for p in state["positions"])
        amount = SIZE * eq
        if made >= MAX_BUYS or invested + amount > CAP * eq + 1e-9 or amount > state["cash"] or c["symbol"] in held:
            continue
        shares = amount * (1 - HALF_COST) / c["price"]
        state["cash"] -= amount
        state["positions"].append({"symbol": c["symbol"], "entry_date": today, "entry_price": c["price"], "shares": shares,
                                   "high": c["price"], "spy_entry": spy_price, "why": c.get("why", ""),
                                   "vs_spy_drop_pct": round(c["vs_spy"] * 100, 2)})
        held.add(c["symbol"])
        made += 1
        bought.append(c)
    state["buys"][today] = made
    state["buys"] = {d: n for d, n in state["buys"].items() if d >= today}      # only today's count matters
    return bought


def run(cfg, broker, market) -> int:
    from .main import (BIG_STOCKS, LIQUID_MIN_DOLLAR_VOLUME_M, MIN_DOLLAR_VOLUME_M, MIN_SMALL_STOCK_DOLLAR_VOLUME_M,
                       MIN_SMALL_STOCK_PRICE, NEWS_MAX_VOLUME_RATIO, NEWS_MIN_ATR_PCT, _heavy_volume)
    from . import earnings_calendar
    from .indicators import summarize

    if not broker.market_open():
        log.info("market closed; nothing to do")
        return 0
    today = datetime.now(ET).strftime("%Y-%m-%d")
    headlines = market.news(hours=2, limit=50)
    from .name_check import filter_headlines
    lookup = getattr(broker, "asset", lambda _s: None)
    headlines, _ = filter_headlines(headlines, {s: getattr(lookup(s), "name", None)
                                                for s in {s for h in headlines for s in h["symbols"]}})
    first = {}
    for h in headlines:
        for s in h["symbols"]:
            if s.isalpha() and len(s) <= 5:
                first.setdefault(s, h["headline"])
    state = journal.read(FILE, None)
    held = [p["symbol"] for p in (state or {}).get("positions", [])]
    symbols = sorted(set(first) | set(held) | {"SPY"})[:60]
    quotes = market.quotes(symbols)
    if "SPY" not in quotes:
        log.warning("no SPY quote; skipping")
        return 0
    spy_px, spy_move = quotes["SPY"].price, quotes["SPY"].price / quotes["SPY"].prev_close - 1
    if state is None:
        state = new_state(spy_px, today)
    prices = {s: q.price for s, q in quotes.items()}
    closed = apply_exits(state, prices, spy_px, today)
    candidates, calendar, checked = [], None, False
    for sym, headline in first.items():
        q = quotes.get(sym)
        if q is None or sym == "SPY" or any(p["symbol"] == sym for p in state["positions"]):
            continue
        vs = q.price / q.prev_close - 1 - spy_move
        if vs > -MIN_DROP:
            continue
        ok, lev, kind, _ = broker.vet(sym)
        if not ok or lev or kind not in ("big", "small"):
            continue
        df = market.daily_bars([sym]).get(sym)
        if df is None or len(df) < 30:
            continue
        t = summarize(df)
        if kind == "small" and t["price"] < MIN_SMALL_STOCK_PRICE:
            continue
        if t["avg_dollar_vol_m"] < (MIN_SMALL_STOCK_DOLLAR_VOLUME_M if kind == "small" else MIN_DOLLAR_VOLUME_M):
            continue
        if (t.get("atr_pct") or 0) < NEWS_MIN_ATR_PCT:
            continue
        if sym not in BIG_STOCKS and t["avg_dollar_vol_m"] < LIQUID_MIN_DOLLAR_VOLUME_M:
            continue
        vr = _heavy_volume(df)
        if vr is not None and vr >= NEWS_MAX_VOLUME_RATIO:
            continue
        if not checked:
            calendar, checked = earnings_calendar.upcoming(today, cfg.earnings_blackout_days), True
        if calendar is None or sym in calendar:        # unknown or close: skip, like the real hunter
            continue
        candidates.append({"symbol": sym, "vs_spy": vs, "price": q.price, "why": headline})
    bought = pick_buys(state, candidates, prices, spy_px, today)
    eq = equity(state, prices)
    state["curve"].append({"t": datetime.now(ET).isoformat(timespec="minutes"), "equity": round(eq, 2), "spy": round(spy_px, 2)})
    state["curve"] = state["curve"][-CURVE_KEEP:]
    state["updated"] = state["curve"][-1]["t"]
    journal.write(FILE, state)
    log.info("paper twin: equity $%.2f; bought %s; closed %s", eq, [c["symbol"] for c in bought], [c["symbol"] for c in closed])
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .broker import Broker
    from .config import load_config
    from .market import Market

    cfg = load_config()
    return run(cfg, Broker(cfg), Market(cfg))


if __name__ == "__main__":
    sys.exit(main())
