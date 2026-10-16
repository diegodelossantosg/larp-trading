"""Buy after strong earnings? Tested on four years of real earnings headlines.

    python -m trader.earnings_backtest --months 48

Finds each quarterly results headline about the ~60 big companies (Benzinga style: "... Q3 EPS
$1.57 Beats $1.43 Estimate ..."), takes the first trading day the market could react, and measures
that day's move vs SPY (the "earnings reaction"). We buy at that day's close and try every
keep-days / stop combination. Results are grouped by how strong the reaction was, and by whether
the headline said the earnings beat or missed.

Known effect from research ("post-earnings drift"): big positive surprises tend to keep drifting up
for weeks. This checks whether it still pays, after costs, on these companies and these years.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, time, timezone

import pandas as pd

from . import exit_backtest as eb
from .news_backtest import fetch_news
from .universe import BIG_STOCKS

log = logging.getLogger("earnings")

EPS = re.compile(r"\bEPS\b")
SURPRISE = re.compile(r"\b(beats?|miss(es)?|in[- ]line|estimate)\b", re.I)
BEAT = re.compile(r"EPS[^,;]*\bbeats?\b", re.I)
MISS = re.compile(r"EPS[^,;]*\bmiss(es)?\b", re.I)
GROUPS = [("fell more than 5%", -1, -0.05), ("fell 2% to 5%", -0.05, -0.02), ("moved less than 2%", -0.02, 0.02),
          ("rose 2% to 5%", 0.02, 0.05), ("rose more than 5%", 0.05, 1)]
HOLDS = (1, 5, 10, 20, 40)


def earnings_events(news: list[dict], daily: dict[str, eb.Daily], spy: eb.Daily) -> list[dict]:
    """One event per company per quarter: the first results headline, with the next session's reaction."""
    last: dict[str, pd.Timestamp] = {}
    out = []
    for n in sorted(news, key=lambda x: x["time"]):
        h = n["headline"]
        if not (EPS.search(h) and SURPRISE.search(h)) or len(n["symbols"]) > 2:
            continue
        ts = pd.Timestamp(n["time"]).tz_convert("America/New_York")
        for sym in n["symbols"]:
            if sym not in BIG_STOCKS or sym not in daily:
                continue
            if sym in last and (ts - last[sym]).days < 45:
                continue
            last[sym] = ts
            px = daily[sym]
            day = ts.date() if ts.time() < time(16, 0) else (ts + pd.Timedelta(days=1)).date()
            i = next((px.pos[d] for d in px.dates if d >= day), None) if day not in px.pos else px.pos[day]
            if i is None or i < 15 or i + 1 >= len(px.c):
                continue
            s = spy.pos.get(px.dates[i])
            if s is None or s == 0:
                continue
            reaction = (px.c[i] / px.c[i - 1] - 1) - (spy.c[s] / spy.c[s - 1] - 1)
            out.append({"symbol": sym, "day": str(px.dates[i]), "i": i, "reaction": float(reaction),
                        "beat": bool(BEAT.search(h)), "miss": bool(MISS.search(h)), "headline": h[:120]})
    return out


def analyze(events: list[dict], daily: dict[str, eb.Daily], spy: eb.Daily) -> dict:
    def sim(e, hold, stop):
        return eb.daily_exit(daily[e["symbol"]], spy, e["i"], hold, stop)

    def block(evs):
        entries = [(int(e["day"][:4]), daily[e["symbol"]].atr_pct[e["i"]], k) for k, e in enumerate(evs)]
        rows = eb.grid(entries, HOLDS, lambda k, h, s: sim(evs[k], h, s))
        by_avg = sorted(rows, key=lambda r: -r["avg_vs_spy_pct"])
        hold20 = [r for r in rows if r["keep_days"] == 20 and r["stop"] in ("no stop", "smart 4x daily swing", "10%")]
        return {"events": len(evs), "best": by_avg[:5], "worst": by_avg[-2:], "keep_20_days": hold20}

    report = {"events": len(events)}
    for label, lo, hi in GROUPS:
        g = [e for e in events if lo <= e["reaction"] < hi]
        report[label] = block(g) if len(g) >= 30 else {"events": len(g)}
    report["headline_says_beat_and_rose_2pct_plus"] = block([e for e in events if e["beat"] and e["reaction"] >= 0.02])
    report["headline_says_miss_and_fell_2pct_plus"] = block([e for e in events if e["miss"] and e["reaction"] <= -0.02])
    report["examples"] = sorted(events, key=lambda e: -e["reaction"])[:8]
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=48)
    ap.add_argument("--out", default="docs/data/earnings_backtest.json")
    args = ap.parse_args()
    symbols = sorted(BIG_STOCKS)
    news = fetch_news(args.months, symbols)
    raw = eb.fetch_daily_ohlc(symbols + ["SPY"], args.months / 12 + 0.3)
    daily = {s: eb.Daily(df) for s, df in raw.items()}
    events = earnings_events(news, daily, daily["SPY"])
    log.info("earnings events: %d", len(events))
    report = analyze(events, daily, daily["SPY"])
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    report["months"] = args.months
    report["headlines_scanned"] = len(news)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
