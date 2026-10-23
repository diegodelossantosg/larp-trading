"""How much does checking the news every 15 minutes beat checking every hour? Measured, not guessed.

    python -m trader.speed_backtest --months 40

Simulates the News Hunter's real loop on 5-minute prices for the same ~265 companies as the event
dataset (today's big names + the 2018 most-traded ranked 101-300), 40 months of real headlines:

  - it looks every P minutes (5, 15, 30, 60) from 9:30 to 16:00 New York time;
  - at each look it sees headlines from the last 2 hours (non-earnings; earnings days left out);
  - if such a company is down 2%+ vs SPY since yesterday's close, it buys 3 minutes later (Claude's
    thinking time) at that 5-minute bar's open, once per company per day;
  - it sells at the close 10 trading days later; result vs SPY over the same time, after 0.1% costs.

Reported per interval: number of buys, average result per buy, and how much that adds up to per
year on a $1,000 portfolio buying 5% ($50) per trade, by company group and by year.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from .event_dataset import EARNINGS_ANY, split_of
from .news_backtest import fetch_news

log = logging.getLogger("speed")

NY = "America/New_York"
INTERVALS = (5, 15, 30, 60)
LAG_MIN = 3
LOOKBACK_MIN = 120
DROP = -0.02
HOLD = 10
COST = 0.001
TRADE_USD = 50.0


class Day:
    """One session of 5-minute bars."""

    def __init__(self, times: np.ndarray, opens: np.ndarray, closes: np.ndarray):
        self.t, self.o, self.c = times, opens, closes      # t: minutes since midnight NY

    def price_at(self, minute: int, field: str = "c") -> float | None:
        """Last close at or before `minute` (field c) or the open of the first bar at/after it (o)."""
        if field == "c":
            k = np.searchsorted(self.t, minute - 5, side="right") - 1   # bar [t, t+5) finished by minute
            return float(self.c[k]) if k >= 0 else None
        k = np.searchsorted(self.t, minute)
        return float(self.o[k]) if k < len(self.t) else None


def sessions(df: pd.DataFrame) -> tuple[list, dict]:
    ny = df.index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute
    keep = (mins >= 570) & (mins < 960)
    df, ny, mins = df[keep], ny[keep], mins[keep]
    days = {}
    for d, idx in pd.Series(range(len(df)), index=ny.date).groupby(level=0):
        k = idx.to_numpy()
        days[d] = Day(mins.to_numpy()[k], df["open"].to_numpy(float)[k], df["close"].to_numpy(float)[k])
    order = sorted(days)
    return order, days


def simulate(news_by_sym: dict, bars: dict, spy_order: list, spy_days: dict, sets: dict, interval: int) -> list[dict]:
    spy_pos = {d: i for i, d in enumerate(spy_order)}
    trades = []
    for sym, items in news_by_sym.items():
        if sym not in bars:
            continue
        order, days = bars[sym]
        pos = {d: i for i, d in enumerate(order)}
        by_day = defaultdict(list)
        for minute, d in items:
            by_day[d].append(minute)
        for d, minutes in by_day.items():
            i, j = pos.get(d), spy_pos.get(d)
            if i is None or j is None or i == 0 or j == 0 or i + HOLD >= len(order) or j + HOLD >= len(spy_order):
                continue
            day, sday = days[d], spy_days[d]
            prev, sprev = days[order[i - 1]].c[-1], spy_days[spy_order[j - 1]].c[-1]
            first = min(minutes)
            look = 570 + ((max(first, 570) - 570) // interval + 1) * interval   # first look after the headline
            while look < 960:
                recent = [m for m in minutes if look - LOOKBACK_MIN <= m < look]
                if recent:
                    p, q = day.price_at(look), sday.price_at(look)
                    if p and q and (p / prev - 1) - (q / sprev - 1) <= DROP:
                        e, se = day.price_at(look + LAG_MIN, "o"), sday.price_at(look + LAG_MIN, "o")
                        if e and se:
                            out = days[order[i + HOLD]].c[-1]
                            sout = spy_days[spy_order[j + HOLD]].c[-1]
                            trades.append({"symbol": sym, "set": sets[sym], "day": str(d), "year": d.year,
                                           "look": look, "minutes_after_news": look - max(m for m in recent),
                                           "r": (out / e - 1) - (sout / se - 1) - COST})
                        break
                look += interval
    return trades


def summarize(trades: list[dict], years: float) -> dict:
    def block(ts):
        if not ts:
            return {"buys": 0}
        rs = [t["r"] for t in ts]
        return {"buys": len(ts), "buys_per_year": round(len(ts) / years, 1),
                "avg_vs_spy_pct": round(statistics.mean(rs) * 100, 3),
                "win_rate": round(sum(r > 0 for r in rs) / len(rs), 3),
                "median_minutes_after_news": int(statistics.median(t["minutes_after_news"] for t in ts)),
                "dollars_per_year_on_50_usd_trades": round(sum(rs) * TRADE_USD / years, 2)}
    out = {"all": block(trades)}
    for s in ("big", "dev", "holdout"):
        out[s] = block([t for t in trades if t["set"] == s])
    out["by_year_avg_pct"] = {y: round(statistics.mean(t["r"] for t in trades if t["year"] == y) * 100, 3)
                              for y in sorted({t["year"] for t in trades})}
    return out


def fetch_5min(symbols: list[str], months: float) -> dict:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(months * 30.5) + 30)
    out = {}
    for k in range(0, len(symbols), 10):
        batch = symbols[k:k + 10]
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                                        start=start, end=end, adjustment=Adjustment.ALL,
                                                        feed=DataFeed.SIP)).df
        except Exception as e:
            log.warning("batch %d: %s", k, e)
            continue
        for sym in df.index.get_level_values("symbol").unique():
            out[sym] = sessions(df.xs(sym, level="symbol").sort_index())
        log.info("5-min bars: %d/%d", min(k + 10, len(symbols)), len(symbols))
    return out


def news_index(news: list[dict], symbols: set[str]) -> dict:
    earn = defaultdict(set)
    for n in news:
        if EARNINGS_ANY.search(n["headline"]):
            d = pd.Timestamp(n["time"]).tz_convert(NY).date()
            for s in n["symbols"]:
                earn[s] |= {d, d - timedelta(days=1), d + timedelta(days=1)}
    out = defaultdict(list)
    for n in news:
        if len(n["symbols"]) > 2 or EARNINGS_ANY.search(n["headline"]):
            continue
        ts = pd.Timestamp(n["time"]).tz_convert(NY)
        minute = ts.hour * 60 + ts.minute
        if not (570 <= minute < 945):
            continue
        for s in n["symbols"]:
            if s in symbols and ts.date() not in earn[s]:
                out[s].append((minute, ts.date()))
    return out


def main() -> None:
    from .audit_backtest import universe_2018
    from .universe import BIG_STOCKS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=40)
    ap.add_argument("--out", default="docs/data/speed_backtest.json")
    args = ap.parse_args()
    big = sorted(BIG_STOCKS)
    fresh = [s for s in universe_2018(top=300)[100:] if s not in BIG_STOCKS]
    sets = {**{s: "big" for s in big}, **{s: split_of(s) for s in fresh}}
    symbols = sorted(sets)
    bars = fetch_5min(symbols + ["SPY"], args.months)
    spy_order, spy_days = bars.pop("SPY")
    news = news_index(fetch_news(args.months, [s for s in symbols if s in bars]), set(bars))
    years = args.months / 12
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "months": args.months,
              "rule": f"drop {DROP:.0%} vs SPY, buy {LAG_MIN} min after the look, hold {HOLD} days, {COST:.1%} costs",
              "by_interval_minutes": {}}
    for p in INTERVALS:
        report["by_interval_minutes"][str(p)] = summarize(simulate(news, bars, spy_order, spy_days, sets, p), years)
        log.info("every %d min: %s", p, report["by_interval_minutes"][str(p)]["all"])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
