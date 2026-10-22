"""Collect every earnings and news-move event, with prices at several entry times, into one file.

    python -m trader.event_dataset --months 40

Built once on GitHub (needs the Alpaca keys); the research loop then works on the file locally,
trying rules, finding why they fail and fixing them, without re-downloading anything.

Companies, in three groups (column `set`):
  - big       the ~65 big companies the agent trades (today's winners: biased upwards)
  - dev       half of the 2018 most-traded companies ranked 101-300 (for designing rules)
  - holdout   the other half: kept aside, looked at ONCE at the end to confirm a rule
The split is by a fixed hash of the symbol, so it never changes between runs.

Events:
  - earnings  the first quarterly-results headline per company per quarter (EPS + beat/miss)
  - news      the first non-earnings headline per company per day with a move of 1%+ vs SPY within
              an hour (both drops and rises are kept; the rules filter them later)

For each event, entry prices (and SPY at the same moment) where the news was already out:
  pre 8:00 (pre-market), open 9:30, 11:00, close, and for news "1 hour after the headline".
Exits at the close 1/5/10/20 trading days after the entry day, gross (no costs: the analysis adds
them, higher in the pre-market), without a stop and with the smart stop (4x daily swing, 8-20%).
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import csv
import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from . import exit_backtest as eb
from .news_backtest import fetch_news

log = logging.getLogger("events")

HOLDS = (1, 5, 10, 20)
NY = "America/New_York"
EPS = re.compile(r"\bEPS\b")
SURPRISE = re.compile(r"\b(beats?|miss(es)?|in[- ]line|estimate)\b", re.I)
BEAT = re.compile(r"EPS[^,;]*\bbeats?\b", re.I)
MISS = re.compile(r"EPS[^,;]*\bmiss(es)?\b", re.I)
SALES_BEAT = re.compile(r"(sales|revenue)[^,;]*\bbeats?\b", re.I)
SALES_MISS = re.compile(r"(sales|revenue)[^,;]*\bmiss(es)?\b", re.I)
GUIDE_UP = re.compile(r"\b(raises|boosts|lifts|above)\b[^,;]*\b(guidance|outlook|forecast)\b|\bguidance\b[^,;]*\babove\b", re.I)
GUIDE_DOWN = re.compile(r"\b(lowers|cuts|below)\b[^,;]*\b(guidance|outlook|forecast)\b|\bguidance\b[^,;]*\bbelow\b", re.I)
EARNINGS_ANY = re.compile(r"\b(EPS|earnings|quarterly results)\b", re.I)
TOPICS = {
    "analyst": r"\b(upgrade|downgrade|price target|initiates coverage|maintains)\w*",
    "legal": r"\b(lawsuit|sues|sued|court|judge|settle\w*|class action|antitrust|verdict)\b",
    "probe": r"\b(probe|investigation|investigat\w+|subpoena|SEC|DOJ|FTC|regulator\w*)\b",
    "health": r"\b(FDA|trial|approval|approves|drug|vaccine|CDC)\b",
    "guidance": r"\b(guidance|outlook|forecast|warns|preannounce\w*)\b",
    "deal": r"\b(acqui\w+|merger|buyout|takeover|deal|stake)\b",
    "people": r"\b(CEO|CFO|chief|resign\w*|steps down|appoint\w*|executive)\b",
    "product": r"\b(launch\w*|recall\w*|outage|product|iPhone|chip|model)\b",
    "policy": r"\b(tariff\w*|sanction\w*|ban|export|government|White House|Trump|China)\b",
    "short_report": r"\b(short seller|short report|fraud|Hindenburg|Muddy Waters)\b",
}
TOPIC_RE = {k: re.compile(v, re.I) for k, v in TOPICS.items()}


def split_of(symbol: str) -> str:
    return "dev" if int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


class Hourly:
    """Hourly bars incl. pre-market: price at a given New York time on a given day."""

    def __init__(self, df: pd.DataFrame):
        ny = df.index.tz_convert(NY)
        self.by_day = defaultdict(list)
        for t, o in zip(ny, df["open"].to_numpy(float)):
            self.by_day[t.date()].append((t.time(), o))

    def at(self, day, when: time, before: time | None = None) -> float | None:
        """Open of the first bar starting at/after `when` (and before `before`) that day."""
        for t, o in self.by_day.get(day, []):
            if t >= when and (before is None or t < before):
                return o
        return None

    def after(self, ts: pd.Timestamp) -> tuple[float, time] | None:
        for t, o in self.by_day.get(ts.date(), []):
            if t >= ts.time():
                return o, t
        return None


def entries(px: eb.Daily, i: int, hr: Hourly | None, spy_hr: Hourly | None, spy: eb.Daily, news_ts: pd.Timestamp):
    """{entry name: (stock price, SPY price, index of the entry day)} for moments after the news."""
    day = px.dates[i]
    s = spy.pos.get(day)
    out = {}
    if s is None:
        return out
    ts_day = news_ts.date()
    before_day = ts_day < day
    t = news_ts.time()
    if hr is not None and spy_hr is not None and (before_day or t < time(8, 0)):
        p, q = hr.at(day, time(8, 0), time(9, 30)), spy_hr.at(day, time(8, 0), time(9, 30))
        if p and q:
            out["pre8"] = (p, q, i)
    if before_day or t < time(9, 30):
        out["open"] = (px.o[i], spy.o[s], i)
    if hr is not None and spy_hr is not None and (before_day or t < time(11, 0)):
        p, q = hr.at(day, time(11, 0)), spy_hr.at(day, time(11, 0))
        if p and q:
            out["h11"] = (p, q, i)
    out["close"] = (px.c[i], spy.c[s], i)
    if i + 1 < len(px.c):
        s1 = spy.pos.get(px.dates[i + 1])
        if s1 is not None:
            out["next_open"] = (px.o[i + 1], spy.o[s1], i + 1)
    return out


def stop_pct(px: eb.Daily, i: int) -> float:
    a = px.atr_pct[i]
    return float(min(0.20, max(0.08, 4 * a / 100))) if a and a > 0 else 0.10


def returns(px: eb.Daily, spy: eb.Daily, entry_px: float, spy_px: float, i: int) -> dict:
    """Gross excess vs SPY at the close h days after entry day i, without and with the smart stop."""
    out = {}
    stop = stop_pct(px, i)
    high = max(entry_px, px.c[i])
    stopped_at = None
    for h in range(1, max(HOLDS) + 1):
        j = i + h
        if j >= len(px.c):
            break
        if stopped_at is None and px.l[j] <= high * (1 - stop):
            stopped_at = (min(high * (1 - stop), px.o[j]), j)
        if stopped_at is None:
            high = max(high, px.h[j])
        if h in HOLDS:
            sj = spy.pos.get(px.dates[j])
            if sj is None:
                continue
            bench = spy.c[sj] / spy_px - 1
            out[f"r{h}"] = round(px.c[j] / entry_px - 1 - bench, 5)
            if stopped_at:
                sk = spy.pos.get(px.dates[stopped_at[1]])
                b2 = spy.c[sk] / spy_px - 1 if sk is not None else bench
                out[f"s{h}"] = round(stopped_at[0] / entry_px - 1 - b2, 5)
            else:
                out[f"s{h}"] = out[f"r{h}"]
    return out


def features(px: eb.Daily, spy: eb.Daily, i: int, vol: np.ndarray) -> dict:
    c = px.c
    s = spy.pos.get(px.dates[i])
    f = {"atr_pct": round(float(px.atr_pct[i]), 2) if px.atr_pct[i] == px.atr_pct[i] else None}
    if i >= 21:
        f["ret20_before"] = round(float(c[i - 1] / c[i - 21] - 1), 4)
    if i >= 200:
        f["above_sma50"] = int(c[i - 1] > np.mean(c[i - 50:i]))
        f["above_sma200"] = int(c[i - 1] > np.mean(c[i - 200:i]))
    if s is not None and s >= 200:
        f["spy_above_sma200"] = int(spy.c[s - 1] > np.mean(spy.c[s - 200:s]))
        f["spy_ret20_before"] = round(float(spy.c[s - 1] / spy.c[s - 21] - 1), 4)
    if i >= 21:
        f["dollar_vol_m"] = round(float(np.mean(c[i - 20:i] * vol[i - 20:i]) / 1e6), 1)
        f["vol_ratio"] = round(float(vol[i] / max(1.0, np.mean(vol[i - 20:i]))), 2)
    return f


def move_at(entry, prev_close, spy_prev):
    p, q, _ = entry
    return round((p / prev_close - 1) - (q / spy_prev - 1), 4)


def build(news, daily, hourly, sets) -> list[dict]:
    spy = daily["SPY"]
    spy_hr = hourly.get("SPY")
    vols = {s: d.vol for s, d in daily.items()}
    earn_days = defaultdict(set)
    for n in news:
        if EARNINGS_ANY.search(n["headline"]):
            for s in n["symbols"]:
                earn_days[s].add(pd.Timestamp(n["time"]).tz_convert(NY).date())
    rows, last_earn, seen_news = [], {}, set()
    for n in sorted(news, key=lambda x: x["time"]):
        h = n["headline"]
        ts = pd.Timestamp(n["time"]).tz_convert(NY)
        if len(n["symbols"]) > 2:
            continue
        is_earn = bool(EPS.search(h) and SURPRISE.search(h))
        for sym in n["symbols"]:
            px = daily.get(sym)
            if px is None or sym == "SPY":
                continue
            # first session that could react: same day if before the close, else the next one
            day = ts.date() if ts.time() < time(16, 0) else (ts + pd.Timedelta(days=1)).date()
            i = px.pos.get(day)
            if i is None:
                k = bisect.bisect_left(px.dates, day)
                i = k if k < len(px.dates) else None
            if i is None or i < 30 or i + 2 >= len(px.c):
                continue
            s = spy.pos.get(px.dates[i])
            if s is None or s < 1:
                continue
            if is_earn:
                if sym in last_earn and (ts - last_earn[sym]).days < 45:
                    continue
                last_earn[sym] = ts
                kind = "earnings"
            else:
                if ts.time() < time(4, 0) or ts.time() >= time(15, 0) or (sym, px.dates[i]) in seen_news:
                    continue
                near = {px.dates[k] for k in range(max(0, i - 1), min(len(px.dates), i + 2))}
                if near & earn_days[sym]:
                    continue
                kind = "news"
            ent = entries(px, i, hourly.get(sym), spy_hr, spy, ts)
            if kind == "news":
                # the move an hour after the headline (pre-market headlines: at the open)
                if time(9, 30) <= ts.time():
                    hr, sh = hourly.get(sym), spy_hr
                    a = hr.after(ts + pd.Timedelta(minutes=60)) if hr else None
                    b = sh.after(ts + pd.Timedelta(minutes=60)) if sh else None
                    if not a or not b or a[1] >= time(16, 0):
                        continue
                    ent["hour_after"] = (a[0], b[0], i)
                    mv = move_at(ent["hour_after"], px.c[i - 1], spy.c[s - 1])
                else:
                    mv = move_at(ent["open"], px.c[i - 1], spy.c[s - 1]) if "open" in ent else None
                if mv is None or abs(mv) < 0.01:
                    continue
                seen_news.add((sym, px.dates[i]))
            row = {"kind": kind, "symbol": sym, "set": sets[sym], "day": str(px.dates[i]), "year": px.dates[i].year,
                   "news_time": ts.strftime("%Y-%m-%d %H:%M"), "headline": h[:140].replace("\n", " "),
                   "beat": int(bool(BEAT.search(h))), "miss": int(bool(MISS.search(h))),
                   "sales_beat": int(bool(SALES_BEAT.search(h))), "sales_miss": int(bool(SALES_MISS.search(h))),
                   "guide_up": int(bool(GUIDE_UP.search(h))), "guide_down": int(bool(GUIDE_DOWN.search(h))),
                   "day_move": round((px.c[i] / px.c[i - 1] - 1) - (spy.c[s] / spy.c[s - 1] - 1), 4),
                   **{f"topic_{k}": int(bool(r.search(h))) for k, r in TOPIC_RE.items()},
                   **features(px, spy, i, vols[sym])}
            if kind == "news":
                row["move_1h"] = mv
            for name, e in ent.items():
                row[f"move_{name}"] = move_at(e, px.c[i - 1], spy.c[s - 1]) if e[2] == i else None
                for k, v in returns(px, spy, e[0], e[1], e[2]).items():
                    row[f"{name}_{k}"] = v
            rows.append(row)
    rows += controls(daily, hourly, sets)
    return rows


def controls(daily, hourly, sets, per_year: int = 25, seed: int = 9) -> list[dict]:
    """Random days of the same companies, entered the same ways: the "did the rule beat chance" baseline."""
    rng = np.random.default_rng(seed)
    spy, spy_hr = daily["SPY"], hourly.get("SPY")
    out = []
    for sym, grp in sorted(sets.items()):
        px = daily.get(sym)
        if px is None or len(px.c) < 260:
            continue
        n = int(per_year * (len(px.c) - 230) / 252)
        for i in sorted(set(rng.integers(230, len(px.c) - 2, max(1, n)).tolist())):
            s = spy.pos.get(px.dates[i])
            if s is None:
                continue
            ts = pd.Timestamp(px.dates[i - 1]).tz_localize(NY) + pd.Timedelta(hours=20)
            row = {"kind": "control", "symbol": sym, "set": grp, "day": str(px.dates[i]), "year": px.dates[i].year,
                   "day_move": round((px.c[i] / px.c[i - 1] - 1) - (spy.c[s] / spy.c[s - 1] - 1), 4),
                   **features(px, spy, i, px.vol)}
            for name, e in entries(px, i, hourly.get(sym), spy_hr, spy, ts).items():
                row[f"move_{name}"] = move_at(e, px.c[i - 1], spy.c[s - 1]) if e[2] == i else None
                for k, v in returns(px, spy, e[0], e[1], e[2]).items():
                    row[f"{name}_{k}"] = v
            out.append(row)
    return out


class DailyV(eb.Daily):
    def __init__(self, df):
        super().__init__(df)
        self.vol = df["volume"].to_numpy(float)


def fetch_hourly(symbols: list[str], years: float) -> dict[str, Hourly]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(years * 365.25))
    out = {}
    for k in range(0, len(symbols), 20):
        batch = symbols[k:k + 20]
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Hour, start=start,
                                                        end=end, adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
        except Exception as e:
            log.warning("hourly batch %d: %s", k, e)
            continue
        if df.empty:
            continue
        for sym in df.index.get_level_values("symbol").unique():
            out[sym] = Hourly(df.xs(sym, level="symbol").sort_index())
        log.info("hourly bars: %d/%d symbols", min(k + 20, len(symbols)), len(symbols))
    return out


def main() -> None:
    from .audit_backtest import universe_2018
    from .universe import BIG_STOCKS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=40)
    ap.add_argument("--out", default="research/events.csv.gz")
    args = ap.parse_args()
    big = sorted(BIG_STOCKS)
    fresh = [s for s in universe_2018(top=300)[100:] if s not in BIG_STOCKS]
    sets = {**{s: "big" for s in big}, **{s: split_of(s) for s in fresh}}
    symbols = sorted(sets)
    years = args.months / 12 + 1.0
    raw = eb.fetch_daily_ohlc(symbols + ["SPY"], years)
    daily = {s: DailyV(df) for s, df in raw.items() if len(df) >= 250}
    hourly = fetch_hourly([s for s in symbols if s in daily] + ["SPY"], args.months / 12 + 0.1)
    news = fetch_news(args.months, [s for s in symbols if s in daily])
    rows = build(news, daily, hourly, sets)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (not k[0].isalpha(), k))
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(int)
    for r in rows:
        by[(r["kind"], r["set"])] += 1
    log.info("events written: %d %s", len(rows), dict(by))


if __name__ == "__main__":
    main()
