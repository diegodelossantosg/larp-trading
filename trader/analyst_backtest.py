"""Buy after an analyst upgrade? Tested on real Alpaca/Benzinga headlines, with the usual luck checks.

    python -m trader.analyst_backtest --months 40

Headline kinds (one event per company per day per kind; days with an earnings headline within a
day are left out, so this measures the analyst call, not the results):
  - upgrade              "Goldman Sachs Upgrades Apple to Buy", "Apple Upgraded to Outperform at ..."
  - downgrade            "... Downgrades ...", "... Downgraded to Neutral ..."
  - new coverage, Buy    "Initiates Coverage On Apple with Buy/Outperform/Overweight"
  - price target raised  "Maintains Buy on Apple, Raises Price Target to $250" (very common)
  - price target cut
Top-tier banks (Goldman, Morgan Stanley, JPMorgan, BofA, Citi, ...) are also shown on their own.

Trades: buy at the close of the first day the market could react (the News Hunter's real timing)
or at the next morning's open; hold 1/5/10/20 days, no stop or our smart stop (4x the daily swing);
measured vs SPY after 0.1% costs. Grouped by the reaction that day (vs SPY).

Luck checks, as for the candles: every year on its own; vs random days of the same companies
(same holds); the same rule on other companies (the 2018 most-traded ranked 101-300, never used to
design anything); and buying at the next open instead of the close.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import defaultdict
from datetime import datetime, time, timezone

import numpy as np
import pandas as pd

from . import exit_backtest as eb
from .news_backtest import fetch_news

log = logging.getLogger("analyst")

KINDS = {
    "upgrade": re.compile(r"\bupgrade[sd]?\b", re.I),
    "downgrade": re.compile(r"\bdowngrade[sd]?\b", re.I),
    "new coverage, Buy": re.compile(r"\binitiat\w*\b.*\b(buy|outperform|overweight|strong buy|accumulate)\b", re.I),
    "price target raised": re.compile(r"\b(raises?|boosts?|lifts?|ups)\b.*\bprice target\b|\bprice target\b.*\b(raised|boosted)\b", re.I),
    "price target cut": re.compile(r"\b(lowers?|cuts?|reduces?|trims?)\b.*\bprice target\b|\bprice target\b.*\b(lowered|cut)\b", re.I),
}
TOP_TIER = re.compile(r"goldman|morgan stanley|jp ?morgan|bank of america|bofa|citi|wells fargo|ubs|barclays|"
                      r"deutsche|jefferies|bernstein|evercore|baird|piper|raymond james|wedbush|bmo|rbc", re.I)
EARNINGS = re.compile(r"\b(EPS|earnings|quarterly results|Q[1-4] (sales|revenue))\b", re.I)
REACTIONS = [("fell", -1, -0.01), ("about flat (-1% to +1%)", -0.01, 0.01), ("rose 1% to 3%", 0.01, 0.03),
             ("rose more than 3%", 0.03, 1)]
HOLDS = (1, 5, 10, 20)


def classify(headline: str) -> str | None:
    """The analyst-call kind of a headline (upgrade/downgrade win over price-target notes)."""
    for kind in ("upgrade", "downgrade", "new coverage, Buy", "price target raised", "price target cut"):
        if KINDS[kind].search(headline):
            return kind
    return None


def events(news: list[dict], daily: dict[str, eb.Daily], spy: eb.Daily) -> list[dict]:
    earn_days = defaultdict(set)
    for n in news:
        if EARNINGS.search(n["headline"]):
            d = pd.Timestamp(n["time"]).tz_convert("America/New_York").date()
            for s in n["symbols"]:
                earn_days[s].add(d)
    seen, out = set(), []
    for n in sorted(news, key=lambda x: x["time"]):
        kind = classify(n["headline"])
        if not kind or len(n["symbols"]) > 2:
            continue
        ts = pd.Timestamp(n["time"]).tz_convert("America/New_York")
        day = ts.date() if ts.time() < time(16, 0) else (ts + pd.Timedelta(days=1)).date()
        for sym in n["symbols"]:
            px = daily.get(sym)
            if px is None:
                continue
            i = px.pos.get(day)
            if i is None:
                i = next((px.pos[d] for d in px.dates if d >= day), None)
            if i is None or i < 20 or i + 1 >= len(px.c):
                continue
            near = {px.dates[k] for k in range(max(0, i - 1), min(len(px.dates), i + 2))}
            if near & earn_days[sym] or (sym, kind, px.dates[i]) in seen:
                continue
            seen.add((sym, kind, px.dates[i]))
            s = spy.pos.get(px.dates[i])
            if s is None or s == 0:
                continue
            reaction = (px.c[i] / px.c[i - 1] - 1) - (spy.c[s] / spy.c[s - 1] - 1)
            out.append({"symbol": sym, "kind": kind, "day": str(px.dates[i]), "i": i, "year": px.dates[i].year,
                        "reaction": float(reaction), "top_tier": bool(TOP_TIER.search(n["headline"])),
                        "headline": n["headline"][:120]})
    return out


def next_open(px: eb.Daily, spy: eb.Daily, i: int, hold: int) -> float | None:
    if i + hold >= len(px.c):
        return None
    s0, s1 = spy.pos.get(px.dates[i + 1]), spy.pos.get(px.dates[i + hold])
    if s0 is None or s1 is None:
        return None
    return float(px.c[i + hold] / px.o[i + 1] - spy.c[s1] / spy.o[s0] - eb.COST)


def stop_for(px: eb.Daily, i: int, kind: str) -> float:
    if kind == "no stop":
        return 9.0
    a = px.atr_pct[i]
    return 4 * a / 100 if a and a > 0 else 9.0


def results(evs, daily, spy, hold, stop, timing="close"):
    out = []
    for e in evs:
        px = daily[e["symbol"]]
        r = (eb.daily_exit(px, spy, e["i"], hold, stop_for(px, e["i"], stop)) if timing == "close"
             else next_open(px, spy, e["i"], hold))
        if r is not None:
            out.append((e["year"], r))
    return out


def describe(tr) -> dict:
    if len(tr) < 10:
        return {"trades": len(tr)}
    xs = [r for _, r in tr]
    by = defaultdict(list)
    for y, r in tr:
        by[y].append(r)
    yearly = {str(y): round(statistics.mean(v) * 100, 2) for y, v in sorted(by.items()) if len(v) >= 5}
    sd = statistics.stdev(xs)
    return {"trades": len(xs), "avg_vs_spy_pct": round(statistics.mean(xs) * 100, 3),
            "win_rate": round(sum(x > 0 for x in xs) / len(xs), 3),
            "t_stat": round(statistics.mean(xs) / (sd / len(xs) ** 0.5), 2) if sd else None,
            "years_positive": f"{sum(v > 0 for v in yearly.values())}/{len(yearly)}",
            "median_year_pct": round(statistics.median(yearly.values()), 2) if yearly else None,
            "by_year_pct": yearly}


def random_days(daily, spy, symbols, hold, stop, n=4000, seed=5) -> float | None:
    rng = np.random.default_rng(seed)
    syms = [s for s in symbols if s in daily]
    xs = []
    for _ in range(n):
        px = daily[syms[rng.integers(len(syms))]]
        i = int(rng.integers(60, max(61, len(px.c) - hold - 1)))
        r = eb.daily_exit(px, spy, i, hold, stop_for(px, i, stop))
        if r is not None:
            xs.append(r)
    return round(statistics.mean(xs) * 100, 3) if xs else None


def analyze(evs, daily, spy, symbols) -> dict:
    rep = {"events": len(evs), "by_kind": {}}
    base = {(h, s): random_days(daily, spy, symbols, h, s) for h in HOLDS for s in ("no stop", "smart 4x")}
    rep["random_days_vs_spy_pct"] = {f"{h}d {s}": v for (h, s), v in base.items()}
    for kind in KINDS:
        ke = [e for e in evs if e["kind"] == kind]
        block = {"events": len(ke), "examples": [e["headline"] for e in ke[:4]]}
        if len(ke) >= 30:
            block["all"] = {f"{h}d {s}": describe(results(ke, daily, spy, h, s)) for h in HOLDS
                            for s in ("no stop", "smart 4x")}
            block["top_tier_banks_10d"] = describe(results([e for e in ke if e["top_tier"]], daily, spy, 10, "smart 4x"))
            block["by_reaction_10d"] = {lbl: describe(results([e for e in ke if lo <= e["reaction"] < hi],
                                                              daily, spy, 10, "smart 4x"))
                                        for lbl, lo, hi in REACTIONS}
            block["buy_next_open_10d"] = describe(results(ke, daily, spy, 10, "smart 4x", timing="open"))
        rep["by_kind"][kind] = block
    return rep


def main() -> None:
    from .audit_backtest import universe_2018
    from .universe import BIG_STOCKS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=40)
    ap.add_argument("--out", default="docs/data/analyst_backtest.json")
    args = ap.parse_args()
    big = sorted(BIG_STOCKS)
    others = [s for s in universe_2018(top=300)[100:] if s not in BIG_STOCKS]
    raw = eb.fetch_daily_ohlc(sorted(set(big) | set(others) | {"SPY"}), args.months / 12 + 0.4)
    daily = {s: eb.Daily(df) for s, df in raw.items() if len(df) >= 100}
    spy = daily["SPY"]
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "months": args.months,
              "cost_round_trip_pct": eb.COST * 100}
    for name, syms in (("big companies (what the News Hunter watches)", big),
                       ("other companies (2018 ranks 101-300, a fresh check)", [s for s in others if s in daily])):
        news = fetch_news(args.months, syms)
        evs = events(news, daily, spy)
        log.info("%s: %d headlines, %d analyst events", name, len(news), len(evs))
        report[name] = {"headlines": len(news), **analyze(evs, daily, spy, syms)}
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
