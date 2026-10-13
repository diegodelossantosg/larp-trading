"""Replay the watcher's buy rules on past 15-minute data and measure the results vs SPY.

    python -m trader.backtest --months 12

For every 15-minute bar of every day it builds the same inputs the live watcher sees, scores them
with the same `watch.score`, and "buys" when the score is high enough (same SPY filter, same
skip-the-first-half-hour rule, at most 4 buys a day, one per ETF per day). Each buy is entered at
the next bar's open and exited by the 6% trailing stop or after a fixed number of trading days,
whichever comes first. Its return is compared with SPY's over exactly the same time.

Not modelled: Opus's approve/reject step, and Claude's daily rebalancing. So this answers one
question only: do these signals, on their own, pick ETFs that beat SPY afterwards?
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone

import pandas as pd

from .indicators import atr_pct, candlestick_patterns
from .universe import CORE_ETFS, LEVERAGED
from .watch import Quote, score

log = logging.getLogger("backtest")


def day_features(daily: pd.DataFrame, day, dates=None) -> dict | None:
    """Daily-trend inputs as they were known before `day` opened."""
    import numpy as np

    dates = dates if dates is not None else np.array([ts.date() for ts in daily.index])
    past = daily.iloc[: int(np.searchsorted(dates, day, side="left"))]
    if len(past) < 50:
        return None
    closes = past["close"]
    return {"prev_close": float(closes.iloc[-1]), "high_20d": float(closes.tail(20).max()),
            "atr_pct": atr_pct(past), "sma20": float(closes.tail(20).mean()), "sma50": float(closes.tail(50).mean())}


def quote_at(feat: dict, bars: pd.DataFrame, k: int) -> Quote:
    """What the watcher would see right after bar k (0-based) of the day completed."""
    done = bars.iloc[: k + 1]
    vol = float(done["volume"].sum())
    vwap = float((done["vwap"] * done["volume"]).sum() / vol) if vol else 0.0
    avg = float(done["volume"].mean())
    return Quote(
        price=float(done["close"].iloc[-1]), prev_close=feat["prev_close"], high_20d=feat["high_20d"],
        atr_pct=feat["atr_pct"], sma20=feat["sma20"], sma50=feat["sma50"], vwap=vwap,
        candles=candlestick_patterns(done.iloc[-6:]) if len(done) >= 3 else [],
        volume_ratio=float(done["volume"].iloc[-1] / avg) if avg else 0.0, bars_today=len(done),
    )


def simulate_exit(path: pd.Series, entry_pos: int, trail: float, max_days: int) -> tuple[int, str]:
    """Index of the exit bar in `path` (closes, all days) and why: trailing stop or time limit."""
    entry_day = path.index[entry_pos].date()
    days_seen, last_day, high = 0, entry_day, path.iloc[entry_pos]
    for i in range(entry_pos + 1, len(path)):
        d = path.index[i].date()
        if d != last_day:
            days_seen, last_day = days_seen + 1, d
            if days_seen > max_days:
                return i - 1, "time"
        price = path.iloc[i]
        high = max(high, price)
        if price <= high * (1 - trail):
            return i, "stop"
    return len(path) - 1, "end of data"


def run(intraday: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame], *, min_scores=(4, 5, 6),
        horizons=(3, 5, 10), trail=0.06, max_buys_per_day=4) -> dict:
    import numpy as np

    spy = intraday["SPY"]
    days = sorted({ts.date() for ts in spy.index})
    signals: dict[int, list[tuple]] = {m: [] for m in min_scores}
    by_day = {sym: {d: g for d, g in df.groupby(df.index.date)} for sym, df in intraday.items()
              if sym not in LEVERAGED and sym in CORE_ETFS and sym in daily}
    daily_dates = {sym: np.array([ts.date() for ts in daily[sym].index]) for sym in by_day}
    for day in days:
        feats, todays = {}, {}
        for sym, groups in by_day.items():
            b = groups.get(day)
            if b is None:
                continue
            f = day_features(daily[sym], day, daily_dates[sym])
            if f is not None and len(b) >= 3:
                feats[sym], todays[sym] = f, b
        if "SPY" not in feats:
            continue
        taken = {m: set() for m in min_scores}
        n_bars = len(todays["SPY"])
        for k in range(1, n_bars - 1):  # bars_today >= 2 ⇒ k >= 1; leave a next bar to enter on
            ts = todays["SPY"].index[k]
            spy_q = quote_at(feats["SPY"], todays["SPY"], k)
            if spy_q.price / spy_q.prev_close - 1 < -0.005:
                continue
            for sym, b in todays.items():
                if sym == "SPY" or k + 1 >= len(b) or b.index[k] != ts:
                    continue
                s, _ = score(quote_at(feats[sym], b, k))
                for m in min_scores:
                    if s >= m and sym not in taken[m] and len(taken[m]) < max_buys_per_day:
                        taken[m].add(sym)
                        signals[m].append((sym, b.index[k + 1], s))

    results = {}
    spy_close = spy["close"]
    for m in min_scores:
        for h in horizons:
            trades = []
            for sym, entry_ts, s in signals[m]:
                path = intraday[sym]["close"]
                pos = path.index.get_loc(entry_ts)
                entry = float(intraday[sym]["open"].iloc[pos])
                exit_pos, why = simulate_exit(path, pos, trail, h)
                exit_ts, exit_px = path.index[exit_pos], float(path.iloc[exit_pos])
                spy_in = float(spy_close.asof(entry_ts))
                spy_out = float(spy_close.asof(exit_ts))
                ret = exit_px / entry - 1
                trades.append({"symbol": sym, "entry": str(entry_ts), "exit": str(exit_ts), "why": why,
                               "ret": ret, "excess": ret - (spy_out / spy_in - 1)})
            ex = [t["excess"] for t in trades]
            results[f"score>={m}, hold<={h}d"] = {
                "trades": len(trades),
                "avg_return_pct": round(statistics.mean([t["ret"] for t in trades]) * 100, 3) if trades else None,
                "avg_vs_spy_pct": round(statistics.mean(ex) * 100, 3) if ex else None,
                "median_vs_spy_pct": round(statistics.median(ex) * 100, 3) if ex else None,
                "beat_spy_rate": round(sum(e > 0 for e in ex) / len(ex), 3) if ex else None,
                "t_stat": round(statistics.mean(ex) / (statistics.stdev(ex) / len(ex) ** 0.5), 2) if len(ex) > 2 and statistics.stdev(ex) > 0 else None,
                "stopped_out_rate": round(sum(t["why"] == "stop" for t in trades) / len(trades), 3) if trades else None,
            }
    return results


def fetch(months: int) -> tuple[dict, dict]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(months * 30.5))
    symbols = sorted(set(CORE_ETFS) - LEVERAGED)

    def get(tf, since, chunk):
        out = {}
        for i in range(0, len(symbols), chunk):
            df = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbols[i:i + chunk], timeframe=tf, start=since, end=end,
                adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
            for sym in df.index.get_level_values("symbol").unique():
                out[sym] = df.xs(sym, level="symbol").sort_index()
            log.info("fetched %s bars for %d/%d symbols", tf, min(i + chunk, len(symbols)), len(symbols))
        return out

    daily = get(TimeFrame.Day, start - timedelta(days=120), 40)
    intraday = get(TimeFrame(15, TimeFrameUnit.Minute), start, 8)
    # Regular session only (09:30-16:00 New York).
    for sym, df in intraday.items():
        ny = df.index.tz_convert("America/New_York")
        mask = [(t.hour, t.minute) >= (9, 30) and t.hour < 16 for t in ny]
        intraday[sym] = df[mask]
    return intraday, daily


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--out", default="docs/data/backtest.json")
    args = ap.parse_args()
    intraday, daily = fetch(args.months)
    results = run(intraday, daily)
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "months": args.months,
              "results": results}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
