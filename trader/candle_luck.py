"""Is the "dragonfly doji in an uptrend" result real, or luck? Plus candlestick patterns on 1-minute candles.

    python -m trader.candle_luck --daily-years 8 --minute-months 6

The candle test (candle_backtest) found one stand-out: a dragonfly doji (a candle with a long lower
wick and almost no body) after a 5-day dip, on a stock above its 50-day average, bought at the close
and held 10 days with our smart stop: +0.8% vs SPY per trade, 8 of 9 years. It was 1 of 384
combinations, so it could be luck. Checks:

1. Other stocks: the 2018 most-traded companies ranked 101-300 (never used in the first test),
   and the ETFs on their own.
2. Nudged definition: 162 variants (body / wick limits, 20/50/100-day average, with or without
   the dip). A real effect stays positive for most of them; a lucky one falls apart.
3. Random comparison: 1,000 random sets of the same size, drawn from (a) the same situation without
   the candle (dip in an uptrend) and (b) any uptrend day. How often does chance do as well?
4. Real-world timing: buy at the next morning's open instead of the close.
5. Other holding periods (5, 10, 20 days).

Plus: the same 16 patterns as candle_backtest on 1-minute candles (last 6 months).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import statistics
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import exit_backtest as eb
from .candle_backtest import ETFS, fetch_intraday, run_intraday

log = logging.getLogger("candle_luck")

CANON = {"body": 0.10, "lower": 0.60, "upper": 0.10, "sma": 50, "dip": True}
GRID = {"body": (0.05, 0.10, 0.15), "lower": (0.5, 0.6, 0.7), "upper": (0.05, 0.10, 0.15),
        "sma": (20, 50, 100), "dip": (True, False)}


def signals(px: eb.Daily, body: float, lower: float, upper: float, sma: int, dip: bool):
    """(candle mask, situation mask without the candle) at each day's close."""
    o, h, l, c = px.o, px.h, px.l, px.c
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = (rng > 0) & (np.abs(c - o) <= body * rng) & ((np.minimum(o, c) - l) >= lower * rng) \
            & ((h - np.maximum(o, c)) <= upper * rng)
        avg = pd.Series(c).rolling(sma).mean().to_numpy()
        up = c > avg
        fell = np.r_[np.full(6, False), c[5:-1] < c[:-6]]
    situation = up & (fell if dip else True)
    situation[:max(sma, 60)] = False
    return shape & situation, situation, up & (np.arange(len(c)) >= max(sma, 60))


def stop_for(px: eb.Daily, i: int) -> float:
    atr = px.atr_pct[i]
    return 4 * atr / 100 if atr and atr > 0 else 9.0


def next_open_trade(px: eb.Daily, spy: eb.Daily, i: int, hold: int) -> float | None:
    """Buy at the next morning's open, sell at the close `hold` days after the signal, vs SPY."""
    if i + hold >= len(px.c):
        return None
    s0, s1 = spy.pos.get(px.dates[i + 1]), spy.pos.get(px.dates[i + hold])
    if s0 is None or s1 is None:
        return None
    return float(px.c[i + hold] / px.o[i + 1] - spy.c[s1] / spy.o[s0] - eb.COST)


def trades(pxs: dict[str, eb.Daily], spy: eb.Daily, params: dict, hold: int = 10, timing: str = "close"):
    out = []   # (year, excess)
    for px in pxs.values():
        mask, _, _ = signals(px, **params)
        for i in np.flatnonzero(mask):
            r = (eb.daily_exit(px, spy, i, hold, stop_for(px, i)) if timing == "close"
                 else next_open_trade(px, spy, i, hold))
            if r is not None:
                out.append((px.dates[i].year, r))
    return out


def describe(tr: list[tuple[int, float]]) -> dict:
    if not tr:
        return {"trades": 0}
    xs = [r for _, r in tr]
    years = {}
    for y, r in tr:
        years.setdefault(y, []).append(r)
    yearly = {str(y): round(statistics.mean(v) * 100, 2) for y, v in sorted(years.items()) if len(v) >= 5}
    return {"trades": len(xs), "avg_vs_spy_pct": round(statistics.mean(xs) * 100, 3),
            "win_rate": round(sum(x > 0 for x in xs) / len(xs), 3),
            "years_positive": f"{sum(v > 0 for v in yearly.values())}/{len(yearly)}", "by_year_pct": yearly}


def random_check(pxs: dict[str, eb.Daily], spy: eb.Daily, observed: float, n: int, which: int,
                 draws: int = 1000, seed: int = 11) -> dict:
    """Mean of `n` random entries from the pool (which=1: dip in an uptrend, 2: any uptrend day)."""
    rng = np.random.default_rng(seed)
    pool = []
    for px in pxs.values():
        masks = signals(px, **CANON)
        for i in np.flatnonzero(masks[which]):
            pool.append((px, i))
    if len(pool) > 40000:   # enough for the comparison, keeps the run short
        pool = [pool[k] for k in rng.choice(len(pool), 40000, replace=False)]
    res = []
    for px, i in pool:
        r = eb.daily_exit(px, spy, i, 10, stop_for(px, i))
        if r is not None:
            res.append(r)
    res = np.array(res)
    means = np.array([res[rng.integers(0, len(res), n)].mean() for _ in range(draws)])
    return {"pool_size": int(len(res)), "pool_avg_vs_spy_pct": round(float(res.mean()) * 100, 3),
            "random_sets_avg_pct": round(float(means.mean()) * 100, 3),
            "share_of_random_sets_as_good": round(float((means >= observed).mean()), 4)}


def run(daily: dict[str, pd.DataFrame], first: list[str], fresh: list[str]) -> dict:
    spy = eb.Daily(daily["SPY"])
    D = {s: eb.Daily(df) for s, df in daily.items() if len(df) >= 300}
    sets = {"first test's stocks": {s: D[s] for s in first if s in D and s not in ETFS},
            "new stocks (2018 ranks 101-300)": {s: D[s] for s in fresh if s in D},
            "ETFs only": {s: D[s] for s in ETFS if s in D and s != "SPY"}}
    report = {"canonical_rule": CANON, "sets": {}}
    for name, pxs in sets.items():
        tr = trades(pxs, spy, CANON)
        rep = {"stocks": len(pxs), "result": describe(tr)}
        if name != "ETFs only" and tr:
            obs = statistics.mean(r for _, r in tr)
            rep["vs_random_same_situation"] = random_check(pxs, spy, obs, len(tr), 1)
            rep["vs_random_uptrend_days"] = random_check(pxs, spy, obs, len(tr), 2)
            rep["buy_next_morning_open"] = describe(trades(pxs, spy, CANON, timing="open"))
            rep["holding_days"] = {str(h): describe(trades(pxs, spy, CANON, hold=h)).get("avg_vs_spy_pct")
                                   for h in (5, 10, 20)}
        report["sets"][name] = rep
        log.info("%s: %s", name, rep["result"])
    variants = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        a = describe(trades(sets["first test's stocks"], spy, p))
        b = describe(trades(sets["new stocks (2018 ranks 101-300)"], spy, p))
        variants.append({**p, "first_stocks_avg_pct": a.get("avg_vs_spy_pct"), "first_trades": a["trades"],
                         "new_stocks_avg_pct": b.get("avg_vs_spy_pct"), "new_trades": b["trades"]})
    ok = [v for v in variants if v["first_trades"] >= 30]
    report["nudged_definitions"] = {
        "variants": len(variants),
        "positive_on_first_stocks": f"{sum((v['first_stocks_avg_pct'] or 0) > 0 for v in ok)}/{len(ok)}",
        "positive_on_new_stocks": f"{sum((v['new_stocks_avg_pct'] or 0) > 0 for v in variants if v['new_trades'] >= 30)}"
                                  f"/{sum(v['new_trades'] >= 30 for v in variants)}",
        "median_first_stocks_pct": round(statistics.median(v["first_stocks_avg_pct"] for v in ok), 3) if ok else None,
        "median_new_stocks_pct": round(statistics.median(v["new_stocks_avg_pct"] for v in variants
                                                         if v["new_trades"] >= 30), 3) if variants else None,
        "all": variants}
    return report


def main() -> None:
    from .audit_backtest import universe_2018
    from .universe import BIG_STOCKS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-years", type=float, default=8)
    ap.add_argument("--minute-months", type=float, default=6)
    ap.add_argument("--out", default="docs/data/candle_luck.json")
    args = ap.parse_args()
    ranked = universe_2018(top=300)
    first = sorted(set(ranked[:100]) | set(BIG_STOCKS))
    fresh = [s for s in ranked[100:] if s not in BIG_STOCKS]
    daily = eb.fetch_daily_ohlc(sorted(set(first) | set(fresh) | set(ETFS)), args.daily_years)
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
              "dragonfly_luck_check": run(daily, first, fresh)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    minute = fetch_intraday(sorted(set(BIG_STOCKS) | set(ETFS)), args.minute_months / 12, minutes=1)
    report["one_minute_candles"] = run_intraday(minute, timeframes={"1 min": 1}, base=1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    d = report["dragonfly_luck_check"]
    print(json.dumps({"sets": d["sets"], "nudged": {k: v for k, v in d["nudged_definitions"].items() if k != "all"},
                      "1min_passing": report["one_minute_candles"]["1 min"]["passing"]}, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
