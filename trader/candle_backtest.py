"""Do candlestick patterns (the TikTok day-trading kind) make money after costs?

    python -m trader.candle_backtest --intraday-years 2 --daily-years 8

16 classic patterns (8 bullish, 8 bearish), each found on 5-minute, 15-minute, 1-hour and daily
candles, then traded the way it is usually taught:

Same-day (5m / 15m / 1h candles; ~60 big companies + 16 main ETFs, 2 years of real bars):
  enter at the next candle's open, then one of
  - "2:1 target": stop just past the pattern's low (high for bearish), target twice the risk,
    anything still open is closed at 16:00 (the usual TikTok rule);
  - "hold 1 hour";  - "hold to the close".
  Bearish patterns are traded short (sell first, buy back later) to see if they predict falls.
Daily candles (the ~100 most-traded companies of 2018, crashed ones included, + big companies +
  ETFs, 8 years): buy at the pattern day's close, hold 1/3/5/10 days with no stop or our smart
  stop (4x the daily swing), measured vs SPY. Bearish daily patterns are checked as sell signals:
  does the stock lag SPY afterwards?

Every trade pays 0.1% in costs. Each pattern is compared with entering at random candles under the
same rules ("edge vs random"). Textbook context filters: high volume, with the bigger trend,
against it. A pattern "passes" only with 100+ trades, a profit after costs, and an edge over random
in BOTH halves of the period (daily: in most years, and a positive median year).
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

log = logging.getLogger("candles")

COST = 0.001
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "SMH", "XBI", "GLD", "TLT"]
BULL = ["hammer", "inverted_hammer", "dragonfly_doji", "bullish_engulfing", "piercing_line", "bullish_harami",
        "morning_star", "three_white_soldiers"]
BEAR = ["shooting_star", "hanging_man", "gravestone_doji", "bearish_engulfing", "dark_cloud_cover", "bearish_harami",
        "evening_star", "three_black_crows"]
TIMEFRAMES = {"5 min": 5, "15 min": 15, "1 hour": 60}
DAILY_HOLDS = (1, 3, 5, 10)
CONTEXTS = ("any", "high volume", "with the trend", "against the trend")


def _sh(x: np.ndarray, k: int) -> np.ndarray:
    """x shifted k bars back in time (x[t-k] at position t)."""
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def detect(o, h, l, c, day: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, int]]:
    """{pattern: (mask at the pattern's last candle, number of candles in it)}.

    Reversal patterns need the move they reverse: bullish ones come after a fall (previous close
    below the close 5 candles earlier), bearish ones after a rise. With `day`, all candles involved
    (pattern + that look-back) must be from the same session.
    """
    n = len(c)
    body, rng = np.abs(c - o), h - l
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    bull, bear = c > o, c < o
    p = {k: (_sh(o, k), _sh(h, k), _sh(l, k), _sh(c, k)) for k in (1, 2)}
    po, ph, pl, pc = p[1]
    pbody, prng = np.abs(pc - po), ph - pl
    o2, h2, l2, c2 = p[2]
    body2, rng2 = np.abs(c2 - o2), h2 - l2
    fell = _sh(c, 1) < _sh(c, 6)
    rose = _sh(c, 1) > _sh(c, 6)
    with np.errstate(invalid="ignore"):
        hammer_shape = (body > 0) & (lower >= 2 * body) & (upper <= 0.5 * body) & (rng > 0)
        inv_shape = (body > 0) & (upper >= 2 * body) & (lower <= 0.5 * body) & (rng > 0)
        doji = (rng > 0) & (body <= 0.1 * rng)
        big_prev_bear = (pc < po) & (pbody >= 0.5 * prng)
        big_prev_bull = (pc > po) & (pbody >= 0.5 * prng)
        small_mid = np.abs(pc - po) <= 0.3 * body2
        m = {
            "hammer": (hammer_shape & fell, 1),
            "inverted_hammer": (inv_shape & fell, 1),
            "dragonfly_doji": (doji & (lower >= 0.6 * rng) & (upper <= 0.1 * rng) & fell, 1),
            "bullish_engulfing": ((pc < po) & bull & (o <= pc) & (c >= po) & (body > pbody) & fell, 2),
            "piercing_line": (big_prev_bear & bull & (o <= pc) & (c > (po + pc) / 2) & (c < po) & fell, 2),
            "bullish_harami": (big_prev_bear & bull & (np.maximum(o, c) <= po) & (np.minimum(o, c) >= pc)
                               & (body < 0.5 * pbody) & fell, 2),
            "morning_star": ((c2 < o2) & (body2 >= 0.5 * rng2) & small_mid & bull & (c > (o2 + c2) / 2)
                             & (_sh(c, 2) < _sh(c, 7)), 3),
            "three_white_soldiers": (bull & (pc > po) & (c2 > o2) & (c > pc) & (pc > c2) & (o >= po) & (o <= pc)
                                     & (po >= o2) & (po <= c2) & (body >= 0.5 * rng) & (pbody >= 0.5 * prng)
                                     & (body2 >= 0.5 * rng2), 3),
            "shooting_star": (inv_shape & rose, 1),
            "hanging_man": (hammer_shape & rose, 1),
            "gravestone_doji": (doji & (upper >= 0.6 * rng) & (lower <= 0.1 * rng) & rose, 1),
            "bearish_engulfing": ((pc > po) & bear & (o >= pc) & (c <= po) & (body > pbody) & rose, 2),
            "dark_cloud_cover": (big_prev_bull & bear & (o >= pc) & (c < (po + pc) / 2) & (c > po) & rose, 2),
            "bearish_harami": (big_prev_bull & bear & (np.maximum(o, c) <= pc) & (np.minimum(o, c) >= po)
                               & (body < 0.5 * pbody) & rose, 2),
            "evening_star": ((c2 > o2) & (body2 >= 0.5 * rng2) & small_mid & bear & (c < (o2 + c2) / 2)
                             & (_sh(c, 2) > _sh(c, 7)), 3),
            "three_black_crows": (bear & (pc < po) & (c2 < o2) & (c < pc) & (pc < c2) & (o <= po) & (o >= pc)
                                  & (po <= o2) & (po >= c2) & (body >= 0.5 * rng) & (pbody >= 0.5 * prng)
                                  & (body2 >= 0.5 * rng2), 3),
        }
    ok = np.ones(n, bool)
    ok[:8] = False
    if day is not None:
        ok &= _sh(day.astype(float), 7) == day
    return {k: (np.nan_to_num(v, nan=0).astype(bool) & ok, nb) for k, (v, nb) in m.items()}


def contexts(c: np.ndarray, v: np.ndarray, bullish: bool) -> dict[str, np.ndarray]:
    sma = pd.Series(c).rolling(50).mean().to_numpy()
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        up = c > sma
        return {"any": np.ones(len(c), bool), "high volume": v > 1.5 * vavg,
                "with the trend": up if bullish else ~up & ~np.isnan(sma),
                "against the trend": (~up & ~np.isnan(sma)) if bullish else up}


# ---------- same-day trades ----------

def intraday_trades(o, h, l, c, day_last: np.ndarray, idx: np.ndarray, stop: np.ndarray, short: bool,
                    hold_bars: int) -> dict[str, np.ndarray]:
    """Net returns of entering at the open of bar idx+1 (same session) for each exit rule."""
    e = idx + 1
    last = day_last[idx]
    sign = -1.0 if short else 1.0
    entry = o[e]
    out = {}
    j_hour = np.minimum(e + hold_bars - 1, last)
    out["hold 1 hour"] = sign * (c[j_hour] / entry - 1) - COST
    out["hold to the close"] = sign * (c[last] / entry - 1) - COST
    risk = (entry - stop) if not short else (stop - entry)
    valid = risk > 0
    target = entry + 2 * risk if not short else entry - 2 * risk
    exit_px = np.full(len(e), np.nan)
    open_ = valid.copy()
    for k in range(0, int((last - e).max(initial=0)) + 1):
        j = e + k
        live = open_ & (j <= last)
        if not live.any():
            break
        jj = np.where(live, j, 0)
        if short:
            hit_stop = live & (h[jj] >= stop)
            hit_tgt = live & ~hit_stop & (l[jj] <= target)
            exit_px[hit_stop] = np.maximum(stop, o[jj])[hit_stop]
            exit_px[hit_tgt] = np.minimum(target, o[jj])[hit_tgt]
        else:
            hit_stop = live & (l[jj] <= stop)
            hit_tgt = live & ~hit_stop & (h[jj] >= target)
            exit_px[hit_stop] = np.minimum(stop, o[jj])[hit_stop]
            exit_px[hit_tgt] = np.maximum(target, o[jj])[hit_tgt]
        open_ &= ~(hit_stop | hit_tgt)
    still = valid & np.isnan(exit_px)
    exit_px[still] = c[last][still]
    r = sign * (exit_px / entry - 1) - COST
    r[~valid] = np.nan
    out["2:1 target"] = r
    return out


def resample(df: pd.DataFrame, minutes: int, base: int = 5) -> pd.DataFrame:
    if minutes == base:
        return df
    ny = df.index.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute - 570
    key = pd.DatetimeIndex(ny.normalize()) + pd.to_timedelta((mins // minutes) * minutes + 570, unit="m")
    g = df.groupby(key)
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()})


def run_intraday(bars: dict[str, pd.DataFrame], seed: int = 7, timeframes: dict[str, int] = TIMEFRAMES,
                 base: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    all_days = sorted({d for df in bars.values() for d in df.index.tz_convert("America/New_York").date})
    split = all_days[len(all_days) // 2]
    report = {"period": f"{all_days[0]} to {all_days[-1]}", "second_half_starts": str(split), "symbols": len(bars)}
    for tf, minutes in timeframes.items():
        res = defaultdict(lambda: defaultdict(list))   # (pattern, context, exit) -> half -> returns
        for sym, raw in bars.items():
            df = resample(raw, minutes, base)
            if len(df) < 200:
                continue
            ny = df.index.tz_convert("America/New_York") if df.index.tz else df.index
            dates = np.array(ny.date)
            day = pd.factorize(dates)[0]
            n = len(df)
            day_last = pd.Series(np.arange(n)).groupby(day).transform("max").to_numpy()
            o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
            half = np.where(dates >= split, "second half", "first half")
            hold_bars = max(1, 60 // minutes)
            can_enter = np.arange(n) < day_last
            pats = detect(o, h, l, c, day)
            sample = can_enter & (rng.random(n) < 0.05)
            sample[:8] = False
            pats["random (bullish rules)"] = (sample, 1)
            pats["random (bearish rules)"] = (sample, 1)
            ctx = {True: contexts(c, v, True), False: contexts(c, v, False)}
            for name, (mask, nb) in pats.items():
                short = name in BEAR or name == "random (bearish rules)"
                idx = np.flatnonzero(mask & can_enter)
                if not len(idx):
                    continue
                lows = np.min([_sh(l, k)[idx] for k in range(nb)], axis=0)
                highs = np.max([_sh(h, k)[idx] for k in range(nb)], axis=0)
                rets = intraday_trades(o, h, l, c, day_last, idx, highs if short else lows, short, hold_bars)
                for cname, cm in ctx[not short].items():
                    sel = cm[idx]
                    for ex, r in rets.items():
                        for hv in ("first half", "second half"):
                            rr = r[sel & (half[idx] == hv)]
                            res[(name, cname, ex)][hv].extend(rr[~np.isnan(rr)].tolist())
            log.info("%s %s done", tf, sym)
        report[tf] = summarize_intraday(res)
    return report


def summarize_intraday(res) -> dict:
    def stats(xs):
        return (statistics.mean(xs) if xs else None, len(xs))

    base = {}
    for (name, cname, ex), halves in res.items():
        if name.startswith("random"):
            for hv, xs in halves.items():
                base[(name, cname, ex, hv)] = stats(xs)[0]
            allx = halves["first half"] + halves["second half"]
            base[(name, cname, ex, "all")] = stats(allx)[0]
    rows = []
    for (name, cname, ex), halves in res.items():
        if name.startswith("random"):
            continue
        bname = "random (bearish rules)" if name in BEAR else "random (bullish rules)"
        allx = halves["first half"] + halves["second half"]
        if len(allx) < 30:
            continue
        avg = statistics.mean(allx)
        b = base.get((bname, cname, ex, "all")) or 0.0
        edges = {}
        for hv in ("first half", "second half"):
            m, k = stats(halves[hv])
            edges[hv] = round((m - (base.get((bname, cname, ex, hv)) or 0.0)) * 100, 3) if k >= 10 else None
        passes = (len(allx) >= 100 and avg > 0 and all(e is not None and e > 0 for e in edges.values()))
        rows.append({"pattern": name, "side": "sell (short)" if name in BEAR else "buy", "context": cname, "exit": ex,
                     "trades": len(allx), "avg_net_pct": round(avg * 100, 3),
                     "win_rate": round(sum(x > 0 for x in allx) / len(allx), 3),
                     "random_avg_pct": round(b * 100, 3), "edge_vs_random_pct": round((avg - b) * 100, 3),
                     "edge_first_half_pct": edges["first half"], "edge_second_half_pct": edges["second half"],
                     "passes": passes})
    rows.sort(key=lambda r: -r["avg_net_pct"])
    baselines = [{"rules": k[0], "context": k[1], "exit": k[2], "avg_net_pct": round(v * 100, 3)}
                 for k, v in base.items() if k[3] == "all" and k[1] == "any" and v is not None]
    return {"passing": [r for r in rows if r["passes"]], "best": rows[:15], "worst": rows[-5:],
            "plain_patterns": [r for r in rows if r["context"] == "any" and r["exit"] == "2:1 target"],
            "random_baselines": baselines, "combinations_tested": len(rows)}


# ---------- daily candles ----------

def run_daily(daily: dict[str, pd.DataFrame], seed: int = 7) -> dict:
    from . import exit_backtest as eb

    rng = np.random.default_rng(seed)
    spy = eb.Daily(daily["SPY"])
    res = defaultdict(lambda: defaultdict(list))   # (pattern, context, hold, stop) -> year -> excess
    for sym, df in daily.items():
        if sym == "SPY" or len(df) < 300:
            continue
        px = eb.Daily(df)
        o, h, l, c = px.o, px.h, px.l, px.c
        v = df["volume"].to_numpy(float)
        pats = detect(o, h, l, c)
        sample = rng.random(len(c)) < 0.05
        sample[:60] = False
        pats["random days"] = (sample, 1)
        ctx_b, ctx_s = contexts(c, v, True), contexts(c, v, False)
        for name, (mask, _) in pats.items():
            bearish = name in BEAR
            ctx = ctx_s if bearish else ctx_b
            for i in np.flatnonzero(mask):
                if i < 60:
                    continue
                year = px.dates[i].year
                atr = px.atr_pct[i]
                stops = [("no stop", 9.0)] + ([("smart 4x daily swing", 4 * atr / 100)] if atr > 0 else [])
                for hold in DAILY_HOLDS:
                    for slabel, s in stops:
                        if bearish and slabel != "no stop":
                            continue
                        r = eb.daily_exit(px, spy, i, hold, s)
                        if r is None:
                            continue
                        for cname, cm in ctx.items():
                            if cm[i] or name == "random days":
                                res[(name, cname if name != "random days" else "any", hold, slabel)][year].append(r)
        log.info("daily %s done", sym)
    return summarize_daily(res)


def summarize_daily(res) -> dict:
    base = {(h, s): {y: statistics.mean(x) for y, x in yrs.items() if x}
            for (n, cn, h, s), yrs in res.items() if n == "random days"}
    rows = []
    for (name, cname, hold, slabel), yrs in res.items():
        if name == "random days":
            continue
        allx = [x for xs in yrs.values() for x in xs]
        if len(allx) < 30:
            continue
        b = base.get((hold, slabel), {})
        edges = {str(y): round((statistics.mean(xs) - b.get(y, 0.0)) * 100, 3) for y, xs in sorted(yrs.items())
                 if len(xs) >= 10}
        bear = name in BEAR
        # bearish: a useful sell signal means the stock does WORSE than random days afterwards
        good = [(-e if bear else e) for e in edges.values()]
        med = statistics.median(good) if good else None
        avg = statistics.mean(allx)
        base_all = statistics.mean([x for (n, cn, h, s), yrs2 in res.items() if n == "random days" and h == hold
                                    and s == slabel for xs in yrs2.values() for x in xs] or [0])
        passes = (len(allx) >= 100 and med is not None and med > 0 and sum(g > 0 for g in good) >= 0.6 * len(good)
                  and (avg > 0 if not bear else avg < base_all))
        rows.append({"pattern": name, "side": "sell signal" if bear else "buy", "context": cname, "keep_days": hold,
                     "stop": slabel, "trades": len(allx), "avg_vs_spy_pct": round(avg * 100, 3),
                     "random_days_vs_spy_pct": round(base_all * 100, 3),
                     "edge_vs_random_pct": round((avg - base_all) * 100, 3) * (-1 if bear else 1),
                     "median_year_edge_pct": round(med, 3) if med is not None else None,
                     "years_with_edge": f"{sum(g > 0 for g in good)}/{len(good)}", "edge_by_year": edges,
                     "passes": passes})
    rows.sort(key=lambda r: -(r["median_year_edge_pct"] or -99))
    return {"passing": [r for r in rows if r["passes"]], "best": rows[:15], "worst": rows[-5:],
            "plain_patterns": [r for r in rows if r["context"] == "any" and r["keep_days"] == 5
                               and r["stop"] == "no stop"],
            "combinations_tested": len(rows)}


# ---------- data ----------

def fetch_intraday(symbols: list[str], years: float, minutes: int = 5) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(years * 365))
    out = {}
    for sym in symbols:
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                                                        start=start, end=end, adjustment=Adjustment.ALL,
                                                        feed=DataFeed.SIP)).df
        except Exception as e:
            log.warning("%s: %s", sym, e)
            continue
        if df.empty:
            continue
        df = df.xs(sym, level="symbol").sort_index()
        ny = df.index.tz_convert("America/New_York")
        out[sym] = df[[(t.hour, t.minute) >= (9, 30) and t.hour < 16 for t in ny]]
        log.info("%d-min bars %s: %d", minutes, sym, len(out[sym]))
    return out


def main() -> None:
    from .audit_backtest import universe_2018
    from .exit_backtest import fetch_daily_ohlc
    from .universe import BIG_STOCKS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--intraday-years", type=float, default=2)
    ap.add_argument("--daily-years", type=float, default=8)
    ap.add_argument("--out", default="docs/data/candle_backtest.json")
    args = ap.parse_args()
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "cost_round_trip_pct": COST * 100}
    daily_syms = sorted(set(universe_2018()) | set(BIG_STOCKS) | set(ETFS))
    report["daily"] = run_daily(fetch_daily_ohlc(daily_syms, args.daily_years))
    report["daily"]["symbols"] = len(daily_syms)
    report["intraday"] = run_intraday(fetch_intraday(sorted(set(BIG_STOCKS) | set(ETFS)), args.intraday_years))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    short = {k: (v if k not in ("daily", "intraday") else
                 {tf: {"passing": x.get("passing"), "best": x.get("best", [])[:5]} if isinstance(x, dict) else x
                  for tf, x in v.items()}) for k, v in report.items()}
    print(json.dumps(short, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
