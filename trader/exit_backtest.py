"""How long to keep a buy and where to put the stop, plus a luck check of the Guard's one surviving rule.

    python -m trader.exit_backtest --years 8.5

Part A, exits. For three kinds of buys:
  - Guard ETF buys (the current 5-of-6 signals, 15-minute data),
  - company stocks bought after falling more than 2% vs SPY in a day (the News Hunter's kind of buy),
  - monthly trend picks (the strongest trends among ETFs, and among big companies: Claude's kind),
every combination of "keep at most N trading days" and stop is replayed:
  fixed stops 3% to 20% or none, and smart stops = 2x, 3x or 4x the holding's normal daily swing (ATR).
Each trade is compared with SPY over the same time, after 0.1% costs. A combination is judged by how
it did in each calendar year (median year, and how many years beat SPY), not by one lucky period.

Part B, is the surviving Guard rule (uptrend + RSI below 70 + volume rising) luck?
  1. older years it never saw (before Oct 2024),  2. nudged settings,
  3. which ETFs and years carry it,  4. versus random buys of uptrending ETFs at the same moments.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import signal_backtest as sb
from .universe import BIG_STOCKS, CORE_ETFS, LEVERAGED

log = logging.getLogger("exit_backtest")

COST = 0.001
FIXED = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, None)
SMART = (2, 3, 4)
SHORT_HOLDS = (1, 2, 3, 5, 10, 20)
LONG_HOLDS = (5, 10, 20, 40, 60)
AUDIT_START = "2024-10-14"   # the signal audit used data from here on; earlier years are unseen


def stops_for(atr_pct: float) -> list[tuple[str, float]]:
    out = [(f"{int(s * 100)}%" if s else "no stop", s if s else 9.0) for s in FIXED]
    out += [(f"smart {k}x daily swing", k * atr_pct / 100) for k in SMART if atr_pct and atr_pct > 0]
    return out


# ---------- daily-bar exits (stocks and monthly picks) ----------

class Daily:
    def __init__(self, df: pd.DataFrame):
        self.dates = [ts.date() for ts in df.index]
        self.pos = {d: i for i, d in enumerate(self.dates)}
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tr = np.maximum(self.h - self.l, np.maximum(np.abs(self.h - np.r_[np.nan, self.c[:-1]]),
                                                    np.abs(self.l - np.r_[np.nan, self.c[:-1]])))
        self.atr_pct = pd.Series(tr).rolling(14).mean().to_numpy() / self.c * 100


def daily_exit(px: Daily, spy: Daily, i0: int, hold: int, stop: float) -> float | None:
    """Buy at the close of day i0; trailing stop on daily lows; sell at the close after `hold` days."""
    entry, high = px.c[i0], px.c[i0]
    if i0 + hold >= len(px.c):
        return None
    out, j = None, i0
    for n in range(1, hold + 1):
        j = i0 + n
        if px.l[j] <= high * (1 - stop):
            out = min(high * (1 - stop), px.o[j])
            break
        high = max(high, px.h[j])
    if out is None:
        out = px.c[j]
    s0, s1 = spy.pos.get(px.dates[i0]), spy.pos.get(px.dates[j])
    if s0 is None or s1 is None:
        return None
    return float(out / entry - spy.c[s1] / spy.c[s0] - COST)


def grid(entries: list[tuple], holds, simulate) -> list[dict]:
    """entries: (year, atr_pct, key). simulate(key, hold, stop) -> excess return or None."""
    rows = []
    labels = [lbl for lbl, _ in stops_for(1.0)]
    for hold in holds:
        for lbl in labels:
            by_year = defaultdict(list)
            for year, atr, key in entries:
                stop = dict(stops_for(atr)).get(lbl)
                if stop is None:
                    continue
                r = simulate(key, hold, stop)
                if r is not None:
                    by_year[year].append(r)
            allr = [x for v in by_year.values() for x in v]
            if len(allr) < 30:
                continue
            yearly = {str(y): round(statistics.mean(v) * 100, 3) for y, v in sorted(by_year.items()) if len(v) >= 5}
            rows.append({"keep_days": hold, "stop": lbl, "trades": len(allr),
                         "avg_vs_spy_pct": round(statistics.mean(allr) * 100, 3),
                         "median_year_pct": round(statistics.median(yearly.values()), 3) if yearly else None,
                         "years_beating_spy": f"{sum(v > 0 for v in yearly.values())}/{len(yearly)}",
                         "by_year": yearly})
    rows.sort(key=lambda r: (-(r["median_year_pct"] or -99), -r["avg_vs_spy_pct"]))
    return rows


def summarize_grid(rows: list[dict], current: tuple[int, str] | None) -> dict:
    out = {"best": rows[:10], "worst": rows[-3:]}
    if current:
        out["current_setting"] = [r for r in rows if (r["keep_days"], r["stop"]) == current]
    fixed = [r for r in rows if not r["stop"].startswith("smart")]
    smart = [r for r in rows if r["stop"].startswith("smart")]
    out["best_fixed"] = fixed[:1]
    out["best_smart"] = smart[:1]
    return out


# ---------- entries ----------

def drop_entries(stocks: dict[str, Daily], spy: Daily) -> list[tuple]:
    """A big company closes >2% (and <15%) below SPY's move for the day; at most 4 a day."""
    by_day = defaultdict(list)
    for sym, px in stocks.items():
        for i in range(15, len(px.c) - 1):
            s = spy.pos.get(px.dates[i])
            if s is None or s == 0:
                continue
            rel = (px.c[i] / px.c[i - 1] - 1) - (spy.c[s] / spy.c[s - 1] - 1)
            if -0.15 < rel < -0.02:
                by_day[px.dates[i]].append((rel, sym, i))
    out = []
    for d, evs in by_day.items():
        for rel, sym, i in sorted(evs)[:4]:      # the biggest drops first
            out.append((d.year, stocks[sym].atr_pct[i], (sym, i)))
    return out


def trend_entries(pxs: dict[str, Daily], spy: Daily, n_picks: int) -> list[tuple]:
    """Month-end picks: strongest 6-month trend per unit of volatility, above the 200-day average."""
    closes = pd.DataFrame({s: pd.Series(p.c, index=p.dates) for s, p in pxs.items()}).sort_index()
    rets = closes.pct_change()
    score = (closes / closes.shift(126) - 1) / (rets.rolling(126).std() * np.sqrt(252))
    score = score.where(closes > closes.rolling(200).mean())
    month_end = closes.groupby(pd.Index([d.strftime("%Y-%m") for d in closes.index])).tail(1).index
    out = []
    for d in month_end:
        for sym in score.loc[d].dropna().sort_values(ascending=False).index[:n_picks]:
            i = pxs[sym].pos[d]
            out.append((d.year, pxs[sym].atr_pct[i], (sym, i)))
    return out


# ---------- part B: the surviving Guard rule ----------

def survivor_checks(sig, meta, exits: sb.Exits) -> dict:
    col = {n: i for i, n in enumerate(sb.NAMES)}
    up = sig[:, col["uptrend"]]
    old = (meta["day"].astype(str) < AUDIT_START).to_numpy()

    def rule(rsi_max=70.0, vol_min=1.2):
        return up & (meta["rsi"].to_numpy() < rsi_max) & (meta["vol_trend"].to_numpy() > vol_min)

    def stats(ids, hold=5, stop=0.08):
        xs = [r for r in (exits.result(i, hold, stop) for i in ids) if r is not None]
        if len(xs) < 3:
            return {"trades": len(xs)}
        sd = statistics.stdev(xs)
        return {"trades": len(xs), "avg_vs_spy_pct": round(statistics.mean(xs) * 100, 3),
                "beat_spy_rate": round(sum(x > 0 for x in xs) / len(xs), 3),
                "t_stat": round(statistics.mean(xs) / (sd / len(xs) ** 0.5), 2) if sd else None}

    base = rule()
    report = {
        "1_older_unseen_years": {"before_" + AUDIT_START: stats(sb.trades(base & old, meta)),
                                 "from_" + AUDIT_START: stats(sb.trades(base & ~old, meta))},
        "2_nudged_settings_older_years": {},
        "3_by_year_and_etf": {},
        "4_vs_random_uptrend_buys_older_years": {},
    }
    for rsi_max in (65, 70, 75):
        for vol_min in (1.1, 1.2, 1.3):
            ids = sb.trades(rule(rsi_max, vol_min) & old, meta)
            for hold, stop in ((3, 0.06), (5, 0.08), (7, 0.10)):
                report["2_nudged_settings_older_years"][f"RSI<{rsi_max}, volume +{round((vol_min - 1) * 100)}%, keep {hold}d, stop {int(stop * 100)}%"] = \
                    stats(ids, hold, stop).get("avg_vs_spy_pct")
    ids_all = sb.trades(base, meta)
    per_year, per_etf = defaultdict(list), defaultdict(list)
    for i in ids_all:
        r = exits.result(i, 5, 0.08)
        if r is not None:
            per_year[str(meta.at[i, "day"].year)].append(r)
            per_etf[meta.at[i, "sym"]].append(r)
    total = sum(sum(v) for v in per_etf.values())
    top = sorted(per_etf.items(), key=lambda kv: -sum(kv[1]))[:5]
    report["3_by_year_and_etf"] = {
        "avg_vs_spy_pct_by_year": {y: round(statistics.mean(v) * 100, 3) for y, v in sorted(per_year.items())},
        "top_5_etfs_share_of_total_gain": {s: round(sum(v) / total, 2) if total else None for s, v in top},
    }
    # Random: on each day the rule bought n ETFs, buy n random uptrending ETFs at random moments that day.
    rng = np.random.default_rng(7)
    picked = sb.trades(base & old, meta)
    days = meta.loc[picked, "day"].value_counts()
    up_old = np.where(up & old)[0]
    up_days = meta["day"].to_numpy()[up_old]
    rand = []
    for d, n in days.items():
        pool = up_old[up_days == d]
        if len(pool):
            rand.extend(rng.choice(pool, size=min(n, len(pool)), replace=False))
    report["4_vs_random_uptrend_buys_older_years"] = {"rule": stats(picked), "random_uptrend_buys": stats(rand),
                                                       "uptrend_and_rsi_without_volume": stats(sb.trades(up & (meta['rsi'].to_numpy() < 70) & old, meta))}
    return report


# ---------- data ----------

def fetch_daily_ohlc(symbols: list[str], years: float) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(years * 365.25) + 300)
    out = {}
    for i in range(0, len(symbols), 40):
        df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols[i:i + 40], timeframe=TimeFrame.Day,
                                                    start=start, end=end, adjustment=Adjustment.ALL,
                                                    feed=DataFeed.SIP)).df
        for sym in df.index.get_level_values("symbol").unique():
            d = df.xs(sym, level="symbol").sort_index()
            d.index = pd.DatetimeIndex([ts.tz_convert("America/New_York").normalize().tz_localize(None) for ts in d.index])
            out[sym] = d
    return out


def main() -> None:
    from . import backtest

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=8.5)
    ap.add_argument("--out", default="docs/data/exit_backtest.json")
    args = ap.parse_args()
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "years": args.years}

    etfs = sorted(set(CORE_ETFS) - LEVERAGED)
    daily = fetch_daily_ohlc(sorted(set(etfs) | set(BIG_STOCKS) | {"SPY"}), args.years)
    log.info("daily bars for %d symbols", len(daily))
    D = {s: Daily(df) for s, df in daily.items()}
    spy = D["SPY"]

    def sim(pxs):
        return lambda key, hold, stop: daily_exit(pxs[key[0]], spy, key[1], hold, stop)

    stocks = {s: D[s] for s in BIG_STOCKS if s in D}
    etf_d = {s: D[s] for s in etfs if s in D and s != "SPY"}
    report["stocks_after_a_drop"] = summarize_grid(grid(drop_entries(stocks, spy), SHORT_HOLDS, sim(stocks)), (3, "10%"))
    log.info("stocks after a drop: done")
    report["stocks_monthly_trend_picks"] = summarize_grid(grid(trend_entries(stocks, spy, 4), LONG_HOLDS, sim(stocks)), None)
    report["etfs_monthly_trend_picks"] = summarize_grid(grid(trend_entries(etf_d, spy, 6), LONG_HOLDS, sim(etf_d)), None)
    log.info("monthly picks: done")

    months = int(args.years * 12)
    intraday, daily_i = backtest.fetch(months, etfs)
    for s, df in daily_i.items():
        df.index = pd.DatetimeIndex([ts.tz_convert("America/New_York").normalize().tz_localize(None) for ts in df.index])
    for s, df in intraday.items():
        df.index = df.index.tz_convert("America/New_York")
    sig, meta = sb.build(intraday, daily_i)
    exits = sb.Exits(intraday, daily_i, meta)
    log.info("guard signal table: %d rows", len(meta))
    col = {n: i for i, n in enumerate(sb.NAMES)}
    guard = sb.trades(sig[:, [col[n] for n in sb.CURRENT]].sum(1) >= 5, meta)
    entries = [(meta.at[i, "day"].year, meta.at[i, "atr_pct"], int(i)) for i in guard]
    report["guard_etf_buys"] = summarize_grid(grid(entries, SHORT_HOLDS, lambda i, h, s: exits.result(i, h, s)), (3, "6%"))
    report["survivor_luck_checks"] = survivor_checks(sig, meta, exits)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
