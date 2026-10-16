"""Does buying a company about an hour after news has pushed it up still beat the S&P 500?

    python -m trader.news_backtest --months 24

The News Hunter checks once an hour, so it always arrives late to a headline. This replays two
years of real Alpaca/Benzinga headlines about the ~60 big companies:

- the "first reaction" is the stock's move vs SPY from just before the headline to 60 minutes after;
- we "buy" at that point (5-minute bar open) and hold until the close of the same day, 1, 3 or 5
  trading days later; the result is compared with SPY over exactly the same time, after 0.1% costs;
- one event per company per day (its first headline in regular hours, before 15:00 New York).

Events are grouped by the first reaction: the News Hunter wants the "rising but not exploded"
group (+1% to +6% vs SPY). Bad-news and no-reaction groups are shown for comparison.

Opus itself can't be replayed fairly here: it already knows how 2024-2026 turned out, and its web
search would read later articles. This tests the idea it acts on, without its judgment.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .universe import BIG_STOCKS

log = logging.getLogger("news_backtest")

COST = 0.001  # round trip
DELAY_MIN = 60
HORIZONS = {"same day close": 0, "1 day": 1, "3 days": 3, "5 days": 5}
GROUPS = [
    ("fell more than 2% (bad news)", -1.0, -0.02),
    ("barely moved (-2% to +1%)", -0.02, 0.01),
    ("rose 1% to 3%", 0.01, 0.03),
    ("rose 3% to 6%", 0.03, 0.06),
    ("rose more than 6% (already exploded)", 0.06, 1.0),
]


def _index(df: pd.DataFrame) -> dict:
    dates = np.array([t.date() for t in df.index])
    last = {}
    for i, d in enumerate(dates):
        last[d] = i
    return {"days": sorted(last), "last": last}


def events(news: list[dict], bars: dict[str, pd.DataFrame]) -> list[dict]:
    """First in-hours headline per company per day, with the price path around it."""
    spy = bars["SPY"]
    cache = {"SPY": _index(spy)}
    seen, out = set(), []
    for n in sorted(news, key=lambda x: x["time"]):
        ts = pd.Timestamp(n["time"]).tz_convert("America/New_York")
        if not (9 * 60 + 30 <= ts.hour * 60 + ts.minute < 15 * 60) or ts.weekday() >= 5:
            continue
        for sym in n["symbols"]:
            if sym not in BIG_STOCKS or sym not in bars or (sym, ts.date()) in seen:
                continue
            seen.add((sym, ts.date()))
            if sym not in cache:
                cache[sym] = _index(bars[sym])
            ev = _measure(bars[sym], spy, ts, cache[sym], cache["SPY"])
            if ev:
                out.append({"symbol": sym, "time": str(ts), "headline": n["headline"][:120], **ev})
    return out


def _measure(px: pd.DataFrame, spy: pd.DataFrame, ts: pd.Timestamp, pc: dict, sc: dict) -> dict | None:
    import bisect

    idx = px.index
    i0 = idx.searchsorted(ts) - 1                      # last bar before the headline
    i1 = idx.searchsorted(ts + pd.Timedelta(minutes=DELAY_MIN))
    if i0 < 0 or i1 >= len(idx) or idx[i1].date() != ts.date():
        return None
    s0 = spy.index.searchsorted(idx[i0], side="right") - 1
    s1 = spy.index.searchsorted(idx[i1])
    if s0 < 0 or s1 >= len(spy):
        return None
    before, entry = px["close"].iloc[i0], px["open"].iloc[i1]
    spy_before, spy_entry = spy["close"].iloc[s0], spy["open"].iloc[s1]
    out = {"reaction": float((entry / before) - (spy_entry / spy_before)), "day": str(ts.date()),
           "entry_price": float(entry), "spy_entry": float(spy_entry)}
    k = bisect.bisect_left(pc["days"], ts.date())
    for name, h in HORIZONS.items():
        if k + h >= len(pc["days"]):
            return None
        day = pc["days"][k + h]
        if day not in sc["last"]:
            return None
        stock = px["close"].iloc[pc["last"][day]] / entry - 1
        bench = spy["close"].iloc[sc["last"][day]] / spy_entry - 1
        out[name] = float(stock - bench - COST)
    return out


def summarize(evs: list[dict]) -> dict:
    report = {}
    for label, lo, hi in GROUPS:
        g = [e for e in evs if lo <= e["reaction"] < hi]
        row = {"events": len(g)}
        for name in HORIZONS:
            xs = [e[name] for e in g]
            if len(xs) > 2:
                sd = statistics.stdev(xs)
                row[name] = {"avg_vs_spy_pct": round(statistics.mean(xs) * 100, 3),
                             "beat_spy_rate": round(sum(x > 0 for x in xs) / len(xs), 3),
                             "t_stat": round(statistics.mean(xs) / (sd / len(xs) ** 0.5), 2) if sd else None}
        report[label] = row
    return report


def fetch_news(months: float, symbols: list[str]) -> list[dict]:
    """Every Alpaca/Benzinga headline about these symbols over the last `months` months."""
    import requests

    from .config import load_config

    cfg = load_config()
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(months * 30.5))
    headers = {"APCA-API-KEY-ID": cfg.alpaca_api_key, "APCA-API-SECRET-KEY": cfg.alpaca_secret_key}
    news: list[dict] = []
    for i in range(0, len(symbols), 10):
        params = {"symbols": ",".join(symbols[i:i + 10]), "start": start.isoformat(), "end": end.isoformat(),
                  "limit": 50, "sort": "asc", "include_content": "false"}
        while True:
            for attempt in range(5):
                r = requests.get("https://data.alpaca.markets/v1beta1/news", params=params, headers=headers, timeout=30)
                if r.status_code != 429:
                    break
                import time
                time.sleep(2 ** attempt)
            r.raise_for_status()
            body = r.json()
            news += [{"time": a["created_at"], "headline": a["headline"], "symbols": a.get("symbols", [])}
                     for a in body.get("news", [])]
            if not body.get("next_page_token"):
                break
            params["page_token"] = body["next_page_token"]
        log.info("news: %d headlines after %d/%d symbols", len(news), min(i + 10, len(symbols)), len(symbols))
    return news


def fetch(months: int) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from .config import load_config

    cfg = load_config()
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(months * 30.5))
    symbols = sorted(BIG_STOCKS)
    news = fetch_news(months, symbols)

    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    bars = {}
    for sym in symbols + ["SPY"]:
        df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                                    start=start, end=end, adjustment=Adjustment.ALL,
                                                    feed=DataFeed.SIP)).df
        if df.empty:
            continue
        df = df.xs(sym, level="symbol").sort_index()
        df.index = df.index.tz_convert("America/New_York")
        bars[sym] = df[[(t.hour, t.minute) >= (9, 30) and t.hour < 16 for t in df.index]]
        log.info("bars: %s %d", sym, len(bars[sym]))
    return news, bars


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--out", default="docs/data/news_backtest.json")
    args = ap.parse_args()
    news, bars = fetch(args.months)
    evs = events(news, bars)
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "months": args.months,
              "headlines": len(news), "events": len(evs), "delay_minutes": DELAY_MIN, "groups": summarize(evs),
              "examples": sorted([e for e in evs if 0.01 <= e["reaction"] < 0.06], key=lambda e: e["time"])[-15:]}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
